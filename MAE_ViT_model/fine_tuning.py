import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC
import torchmetrics.functional as TMF

from main import create_dataloaders
from spectrogram_mae import MaskedAutoencoderViT
from loss_visualization import LossTracker
from visualization import visualize_anomaly_results, plot_roc_curve, plot_error_maps_by_class, plot_error_distribution, plot_confusion_matrix

# --- New CNN Classifier ---
class AnomalyClassifierCNN(nn.Module):
    def __init__(self, in_channels=1, img_size=(496, 496)):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # Output: 16 x 248 x 248
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # Output: 32 x 124 x 124
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2) # Output: 64 x 62 x 62
        # Calculate flattened size after convolutions and pooling
        flattened_size = 64 * (img_size[0] // 8) * (img_size[1] // 8)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 1) # Output layer for binary classification (logits)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Output logits directly
        return x

# --- PyTorch Lightning Module for Fine-Tuning ---
class FineTuningLightningModule(pl.LightningModule):
    def __init__(
        self,
        mae_model_path: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-5,
        pos_weight: torch.Tensor = None,
        mae_img_size: tuple = (496, 496),
        mae_patch_size: tuple = (16, 16),
        in_chans: int = 1,
        mae_mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight', 'mae_model_path']) # Saves args to hparams attribute
        self.pos_weight_value = pos_weight
        self._mae_model_path = mae_model_path

        # Load Pre-trained MAE Model
        self.mae_model = MaskedAutoencoderViT(
            img_size=self.hparams.mae_img_size,
            patch_size=self.hparams.mae_patch_size,
            mask_ratio=self.hparams.mae_mask_ratio
        )

        self._load_mae_weights(self._mae_model_path)

        # Freeze MAE Model
        for param in self.mae_model.parameters():
            param.requires_grad = False
        self.mae_model.eval()

        # Initialize Classifier
        self.classifier = AnomalyClassifierCNN(img_size=self.hparams.mae_img_size, in_channels=self.hparams.in_chans)

        # Define Loss Function
        # pos_weight handles class imbalance
        if pos_weight is None:
            print("Warning: pos_weight not provided. Using default weight of 1.")
            pos_weight = torch.tensor(1.0) # Default if not calculated/passed

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight_value)

        # Initialize Binary AUROC metric
        self.train_auc = BinaryAUROC()
        self.val_auc = BinaryAUROC()

        # --- Data Storage for Plots ---
        self.val_class_0_recon_error = []
        self.val_class_1_recon_error = []
        self.final_val_preds = []
        self.final_val_labels = []
        self.final_val_class_0_error = []
        self.final_val_class_1_error = []
        self.final_thresholds = []
        self.final_threshold_preds = []

    def _load_mae_weights(self, mae_model_path):
        """Helper to load MAE weights from .pth state_dict or .ckpt."""
        if not os.path.exists(mae_model_path):
            raise FileNotFoundError(f"Pretrained MAE model not found at {mae_model_path}")

        try:
            # Attempt 1: Load if .pth is the state_dict of MaskedAutoencoderViT
            self.mae_model.load_state_dict(torch.load(mae_model_path, weights_only=True))
            print(f"Successfully loaded MAE state_dict from {mae_model_path}")
        except:
            try:
                # Attempt 2: Load if .pth is a LightningModule checkpoint
                mae_lightning_checkpoint = torch.load(mae_model_path, map_location=torch.device('cpu'))
                self.mae_model.load_state_dict(mae_lightning_checkpoint['state_dict'])
                print(f"Successfully loaded MAE state_dict from Lightning checkpoint {mae_model_path}")
            except Exception as e:
                print(f"Could not load MAE weights from {mae_model_path}. Error: {e}")
                raise FileNotFoundError("Failed to load MAE model weights with known methods.")

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs the image from patches.

        Args:
            patches (torch.Tensor): Patches tensor of shape [B, L, Ph, Pw]
                                    where L = num_patches = grid_h * grid_w.

        Returns:
            torch.Tensor: Reconstructed image tensor of shape [B, C, H, W].
        """
        # Get dimensions from hyperparameters saved during __init__
        patch_h, patch_w = self.hparams.mae_patch_size
        img_h, img_w = self.hparams.mae_img_size
        in_chans = self.hparams.in_chans

        # Calculate grid size
        grid_h = img_h // patch_h
        grid_w = img_w // patch_w

        # Check if the number of patches matches grid size
        B, L, Ph, Pw = patches.shape
        if L != grid_h * grid_w:
            raise ValueError(f"Number of patches ({L}) does not match grid size ({grid_h}x{grid_w}={grid_h * grid_w})")
        if Ph != patch_h or Pw != patch_w:
             raise ValueError(f"Input patch dimensions ({Ph}x{Pw}) do not match model patch size ({patch_h}x{patch_w})")

        # Reshape to (B, grid_h, grid_w, Ph, Pw)
        patches = patches.reshape(B, grid_h, grid_w, patch_h, patch_w)
        # Permute to (B, grid_h, Ph, grid_w, Pw) - groups rows of patches together
        patches = patches.permute(0, 1, 3, 2, 4)
        # Reshape to (B, H, grid_w, Pw) - combines patch rows into image height
        patches = patches.reshape(B, img_h, grid_w, patch_w)
        # Reshape to (B, H, W) - combines patch columns into image width
        reconstructed_image = patches.reshape(B, img_h, img_w)
        # Add channel dimension: (B, H, W) -> (B, C, H, W)
        reconstructed_image = reconstructed_image.unsqueeze(1)

        if reconstructed_image.shape != (B, in_chans, img_h, img_w):
             raise ValueError(f"Final image shape mismatch: {reconstructed_image.shape} vs {(B, in_chans, img_h, img_w)}")

        return reconstructed_image

    def _get_reconstruction_and_error(self, x):
        self.mae_model.to(x.device)

        # Get Reconstruction Patches from frozen MAE
        with torch.no_grad(): 
             mae_output = self.mae_model(x)  
             reconstructed_patches = mae_output['reconstruction'] # [B, num_patches, patch_h, patch_w]

        # Unpatchify: Reconstruct the full image from patches 
        if reconstructed_patches.dim() != 4: 
            raise ValueError(f"Expected 4D tensor from MAE reconstruction, got {reconstructed_patches.dim()}D") 

        reconstructed_image = self.unpatchify(reconstructed_patches) # Shape: (B, C, H, W) 

        # Calculate Reconstruction Error 
        reconstruction_error = torch.abs(x - reconstructed_image) # Shape: (B, C, H, W)

        return reconstructed_image, reconstruction_error 

    def forward(self, x):
        _, reconstruction_error = self._get_reconstruction_and_error(x)

        self.classifier.to(x.device)
        logits = self.classifier(reconstruction_error) # Shape: (B, 1)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1) # Ensure target is float and shape (B, 1)
        y_int = y.int()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Calculate accuracy or AUC
        preds = torch.sigmoid(logits)
        self.train_auc.update(preds, y_int)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float().unsqueeze(1) # Ensure target is float and shape (B, 1)
        y_int = y.int()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)

        # --- Get Reconstruction Error ---
        _, reconstruction_error_batch = self._get_reconstruction_and_error(x) 
        # --- Calculate & Store Summed Reconstruction Error per Sample ---
        for i in range(x.shape[0]):
            sample_error_sum = torch.sum(reconstruction_error_batch[i]).item()
            self.final_thresholds.append(sample_error_sum)
            if y_int[i].item() == 0:
                self.val_class_0_recon_error.append(sample_error_sum)
            else:
                self.val_class_1_recon_error.append(sample_error_sum)

        # Calculate accuracy or AUC
        preds = torch.sigmoid(logits)
        self.val_auc.update(preds, y_int)
        # Log val_auc per epoch
        self.log('val_auc', self.val_auc, on_epoch=True, prog_bar=True, logger=True)

        # Store predictions and labels for final evaluation plots
        self.final_val_preds.append(preds.detach().cpu())
        self.final_val_labels.append(y_int.detach().cpu())
        return loss
    
    def on_validation_epoch_end(self):
        if self.val_class_0_recon_error:
             avg_class_0_error = sum(self.val_class_0_recon_error) / len(self.val_class_0_recon_error)
             self.log('val_avg_recon_error_class_0', avg_class_0_error, logger=True)

        if self.val_class_1_recon_error:
             avg_class_1_error = sum(self.val_class_1_recon_error) / len(self.val_class_1_recon_error)
             self.log('val_avg_recon_error_class_1', avg_class_1_error, logger=True)

        # Store errors from this epoch for potential later use (like final distribution plot)
        self.final_val_class_0_error.extend(self.val_class_0_recon_error)
        self.final_val_class_1_error.extend(self.val_class_1_recon_error)

        # Clear epoch-specific lists
        self.val_class_0_recon_error.clear()
        self.val_class_1_recon_error.clear()

    def run_final_evaluation_pass(self, dataloader, device):
        """Runs a separate pass to collect data for final evaluation plots."""
        self.eval()
        self.to(device)
        self.final_val_preds = []
        self.final_val_labels = []
        self.final_thresholds = []
        self.final_val_class_0_error.clear()
        self.final_val_class_1_error.clear()

        print("Running final evaluation pass over the dataset...")
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                x = x.to(device)
                y_int = y.int()
                logits = self(x)
                preds = torch.sigmoid(logits)
                _, reconstruction_error_batch = self._get_reconstruction_and_error(x)

                self.final_val_preds.append(preds.cpu())
                self.final_val_labels.append(y_int.cpu())

                for i in range(x.shape[0]):
                    sample_error_sum = torch.sum(reconstruction_error_batch[i]).item()
                    self.final_thresholds.append(sample_error_sum)
                    if y_int[i].item() == 0:
                        self.final_val_class_0_error.append(sample_error_sum)
                    else:
                        self.final_val_class_1_error.append(sample_error_sum)

        print("Final evaluation pass complete.")

        self.final_val_preds = torch.cat(self.final_val_preds) if self.final_val_preds else torch.empty(0)
        self.final_val_labels = torch.cat(self.final_val_labels) if self.final_val_labels else torch.empty(0)

    def configure_optimizers(self):
        # Optimize only the classifier parameters
        optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # Add a learning rate scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {"optimizer": optimizer} #, "lr_scheduler": scheduler, "monitor": "val_loss"}

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Fine-tuning Anomaly Detection with MAE")
    # Arguments for Mode
    parser.add_argument('--mode', type=str, choices=['train', 'validate'], default='train', help="Mode: 'train' for training, 'validate' for validation")
    # Arguments for choosing versions
    parser.add_argument('--lmversion',      type=int,   default=13,     help="Version of the pre-trained MAE model to load (default: 13)")
    parser.add_argument('--smversion',      type=int,   default=None,   help="Version for saving the fine-tuned model (default: None. None finds the next available version)")
    parser.add_argument('--cmversion',      type=int,   default=None,   help="Version for the custom model (default: None)")
    # Arguments for LightningModule and Trainer
    parser.add_argument('--lr',             type=float, default=5e-5,   help='Learning rate for classifier')
    parser.add_argument('--weight_decay',   type=float, default=1e-5,   help='Weight decay for classifier')
    parser.add_argument('--batch_size',     type=int,   default=32,     help='Batch size')
    parser.add_argument('--max_epochs',     type=int,   default=200,    help='Maximum number of epochs for fine-tuning')
    parser.add_argument('--patience',       type=int,   default=20,     help='Patience for early stopping')
    parser.add_argument('--num_workers',    type=int,   default=4,      help='Number of dataloader workers')
    parser.add_argument('--pos_weight',     type=float, default=None,   help='Positive weight for BCE loss (default: None)')
    # Data and Model Paths
    parser.add_argument('--data_dir',       type=str,   default=r'C:\Path\to\data',         help='Directory containing fine-tuning data (subfolders as classes)')
    parser.add_argument('--mae_model_dir',  type=str,   default='C:/Path/to/pre/trained/MAE/ViT/model',  help='Base directory where MAE models are saved (use {} for version)')
    parser.add_argument('--mae_model_name', type=str,   default='mae_model_{}.pth', help='Filename pattern for MAE model state_dict .pth file (use {} for version)')
    parser.add_argument('--save_dir',       type=str,   default='C:/Path/to/where/plots/will/be/saved',  help='Directory to save fine-tuned models and logs')
    parser.add_argument('--classifier_path',type=str,   default='C:/Path/to/previous/trained/classifier',    help="Path to the pre-trained classifier model")
    # Visualization arguments
    parser.add_argument('--threshold',      type=float, default=0.5,    help='Threshold for classification')
    parser.add_argument('--cm_threshold',   type=float, default=6200,   help='Threshold for confusion matrix')
    args = parser.parse_args()

    # --- Determine version number ---
    base_save_dir = args.save_dir
    version_number = None
    clf_version_number = None
    if args.smversion is not None:
        # Use the version provided by the user
        version_number = args.smversion
        print(f"--- Using specified save model version: {version_number} ---")
    else:
        # Automatically determine the next available version
        print(f"--- No save model version specified, determining next available version ---")
        os.makedirs(base_save_dir, exist_ok=True) # Ensure base directory exists
        existing_versions = []
        if os.path.exists(base_save_dir) and os.path.isdir(base_save_dir):
            for item in os.listdir(base_save_dir):
                if item.startswith("version_") and os.path.isdir(os.path.join(base_save_dir, item)):
                    try:
                        version_num = int(item.split("_")[1])
                        existing_versions.append(version_num)
                    except (IndexError, ValueError):
                        continue # Ignore malformed version directories

        if not existing_versions:
            version_number = 0
        else:
            version_number = max(existing_versions) + 1
        print(f"--- Determined next available version: {version_number} ---")
    
    if args.cmversion is not None and args.mode == 'validate':
        clf_version_number = args.cmversion
        print(f"--- Using specified custom model version: {clf_version_number} ---")

    # --- Configuration ---
    torch.set_float32_matmul_precision('high')
    img_size = (496, 496) # Should match MAE pre-training
    patch_size = (16, 16) # Should match MAE pre-training

    # Construct paths
    mae_model_load_path = os.path.join(args.mae_model_dir.format(args.lmversion), args.mae_model_name.format(args.lmversion))
    save_path = os.path.join(args.save_dir, f'version_{version_number}')
    checkpoint_dir = os.path.join(save_path, 'checkpoints')
    log_dir = os.path.join(save_path, 'logs')
    final_model_save_path = os.path.join(save_path, f'final_anomaly_classifier_{version_number}.ckpt')
    classifier_save_path = os.path.join(save_path, f'anomaly_classifier_{version_number}.pth')

    if args.mode == 'validate':
        classifier_load_path = args.classifier_path.format(clf_version_number, clf_version_number)
        print(f"validating classifier from path: {classifier_load_path}")

    os.makedirs(save_path, exist_ok=True)
    if args.mode == 'train':
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    # Use a temporary loader to calculate pos_weight if needed for training
    temp_train_loader_for_count, val_loader = create_dataloaders(
        args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # --- Calculate pos_weight (only for training) ---
    pos_weight_tensor = torch.tensor(1.0) # Default
    if args.mode == 'train' and args.pos_weight is None:
        print("Calculating pos_weight for training...")
        num_negatives = 0
        num_positives = 0
        for _, labels in temp_train_loader_for_count:
            num_positives += torch.sum(labels == 1).item()
            num_negatives += torch.sum(labels == 0).item()
        if num_positives > 0:
            pos_weight_value = num_negatives / num_positives
            pos_weight_tensor = torch.tensor(pos_weight_value)
            args.pos_weight = pos_weight_value
            print(f"Calculated pos_weight: {pos_weight_value:.4f} ({num_negatives} negatives / {num_positives} positives)")
        else:
            print("Warning: No positive samples found in training data. Using default pos_weight 1.0.")
    elif args.mode == 'train' and args.pos_weight is not None:
        pos_weight_tensor = torch.tensor(args.pos_weight)
        print(f"Using provided pos_weight: {args.pos_weight}")

     # Create the actual train_loader with shuffling if training
    train_loader, _ = create_dataloaders(
        args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    del temp_train_loader_for_count

    # --- Initialize Lightning Module ---
    try:
        model = FineTuningLightningModule(
            mae_model_path=mae_model_load_path,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            pos_weight=pos_weight_tensor if args.mode == 'train' else torch.tensor(1.0),
            mae_img_size=img_size,
            mae_patch_size=patch_size
        )

    except Exception as e:
        print(f"Error initializing Lightning Module or loading MAE weights: {e}")
        exit(1)

    if args.mode == 'validate':
        if os.path.exists(classifier_load_path):
            try:
                classifier_state_dict = torch.load(classifier_load_path, map_location='cpu')
                # Handle potential nesting if saved differently
                if 'state_dict' in classifier_state_dict:
                    classifier_state_dict = classifier_state_dict['state_dict']
                # Remove prefix if saved from Lightning module directly
                if list(classifier_state_dict.keys())[0].startswith('classifier.'):
                     classifier_state_dict = {k.replace('classifier.', ''): v for k, v in classifier_state_dict.items()}

                missing_keys, unexpected_keys = model.classifier.load_state_dict(classifier_state_dict, strict=False)
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in classifier state_dict: {unexpected_keys}")
                if missing_keys:
                    print(f"Warning: Missing keys in classifier state_dict: {missing_keys}")
                print("Successfully loaded classifier weights.")
            except Exception as e:
                print(f"Error loading classifier weights from {classifier_load_path}: {e}")
                print("Proceeding with initialized classifier weights.")
        else:
            print(f"Error: Classifier weights file not found at {classifier_load_path}")
            if args.mode == 'validate':
                 print("Validation will proceed with randomly initialized classifier.")

    callbacks = []
    logger = None
    loss_tracker = None
    if args.mode == 'train':
        # --- Callbacks ---
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-anomaly-{epoch:02d}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.0,
            patience=args.patience,
            verbose=True,
            mode='min'
        )
        
        loss_tracker = LossTracker()

        callbacks = [checkpoint_callback, early_stop_callback, loss_tracker]

        # --- Logger ---
        logger = TensorBoardLogger(save_dir=log_dir, name='anomaly_detection')

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.max_epochs if args.mode == 'train' else 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=logger,
        precision='32-true',
        enable_progress_bar=True
    )

    loss_plot_save_path = os.path.join(save_path, f'loss_curves_version_{version_number}.png')
    if args.mode == 'train':
        print(f"\n--- Starting Training (Version {version_number} for {args.max_epochs} epochs) ---")
    # --- Train the model ---
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        print("--- Training Finished ---")

        # Save final Lightning checkpoint (contains trained classifier + MAE)
        final_ckpt_path = os.path.join(save_path, f'final_model_version_{version_number}.ckpt')
        trainer.save_checkpoint(final_ckpt_path)
        print(f"Full final Lightning checkpoint saved to {final_ckpt_path}")

        # Save just the classifier state_dict explicitly
        model_cpu = model.to('cpu')
        torch.save(model_cpu.classifier.state_dict(), classifier_save_path)
        print(f"Final classifier state_dict saved to {classifier_save_path}")

        print("\n--- Running Post-Training Evaluation and Plots ---")
        model.run_final_evaluation_pass(val_loader, device=str(trainer.strategy.root_device))
        if loss_tracker: # Check if loss_tracker was initialized
             loss_tracker.visualize_losses(save_path=loss_plot_save_path)
    
    elif args.mode == 'validate':
        print(f"\n--- Starting Validation Only (Version {version_number}) ---")
        trainer.validate(model, dataloaders=val_loader)
        print("--- Validation Finished ---")
        print("\n--- Running Evaluation and Plots ---")
        model.run_final_evaluation_pass(val_loader, device=str(trainer.strategy.root_device))
    
    """
    ##############################################################################
    TRAINING IS COMPLETE FROM THE CODE ABOVE
    THE CODE BELOW IS ONLY USED FOR SAVING DATA AND VISUALIZATION IN THE PLOTS
    ##############################################################################
    """
    print("--- Running Post-Training Plots ---")

    eval_device = str(trainer.strategy.root_device)
    model.to(eval_device)
    model.eval()
    
    # Define save path for the different plots
    visualization_save_path = os.path.join(save_path, f'anomaly_visualization_version_{version_number}.png')
    roc_plot_save_path = os.path.join(save_path, f'roc_curve_version_{version_number}.png')
    roc_plot_save_path_threshold = os.path.join(save_path, f'roc_curve_error_map_version_{version_number}.png')
    error_map_save_path = os.path.join(save_path, f'error_map_comparison_version_{version_number}.png')
    error_dist_save_path = os.path.join(save_path, f'error_distribution_version_{version_number}.png')
    cm_save_path = os.path.join(save_path, f'confusion_matrix_version_{version_number}.png')
    cm_threshold_save_path = os.path.join(save_path, f'confusion_matrix_error_map_version_{version_number}.png')

    # ---------------------------------
    # Generate Anomaly Visualization
    # ---------------------------------
    print("--- Anomaly Visualizations ---")
    visualize_anomaly_results(
        model=model, # Pass the trained model object
        dataloader=val_loader,
        save_path=visualization_save_path,
        num_images=5,
        device=str(model.device),
        threshold=args.threshold,
    )

    # ---------------------------------
    # Generate X amount of error maps in plot to visualize differences between classes
    # ---------------------------------
    print("--- Generating Error Maps by Class ---")
    plot_error_maps_by_class(
        model=model,
        dataloader=train_loader,
        num_samples_per_class=5,
        save_path=error_map_save_path
    )

    # ---------------------------------
    # Generating loss curves
    # ---------------------------------
    if args.mode == "train":
        print("--- Generating Loss Curves ---")
        loss_tracker.visualize_losses(save_path=loss_plot_save_path)

    print("--- Generating Error Distribution Plot ---")

    # --------------------------------- 
    # Generate Error Distribution Plot 
    # --------------------------------- 
    # Run the final evaluation pass to collect all errors
    model.run_final_evaluation_pass(val_loader, str(model.device))

    # Call the plotting function with the collected errors
    plot_error_distribution(
        errors_class_0=model.final_val_class_0_error,
        errors_class_1=model.final_val_class_1_error,
        save_path=error_dist_save_path
    )

    # Save Collected Error values into a text file
    error_values_save_path = os.path.join(save_path, f'error_values_version_{version_number}.txt')
    try:
        with open(error_values_save_path, 'w') as f:
            f.write("Class 0 Errors:\n")
            for error in model.final_val_class_0_error:
                f.write(f"{error}\n")
            f.write("\nClass 1 Errors:\n")
            for error in model.final_val_class_1_error:
                f.write(f"{error}\n")
            f.write("\nMean Class 0 Error: {}\n".format(sum(model.final_val_class_0_error) / len(model.final_val_class_0_error)))
            f.write("\nMean Class 1 Error: {}\n".format(sum(model.final_val_class_1_error) / len(model.final_val_class_1_error)))
        print(f"Error values saved successfully to {error_values_save_path}")
    except Exception as e:
        print(f"Error saving error values to file: {e}")
    

    # ---------------------------------
    # Generate Confusion Matrix
    # ---------------------------------
    plot_confusion_matrix(
        true_labels=model.final_val_labels,
        pred_labels=model.final_val_preds,
        save_path=cm_save_path,
        class_names=['Normal', 'Anomaly']
    )

    # ---------------------------------
    # Generate ROC AUC Curve and Confusion Matrix
    # ---------------------------------
    print("--- Generating Post-Training ROC Curve and Confusion Matrix ---")
    # ROC Curve and Confusion Matrix
    if model.final_val_preds is not None and model.final_val_labels is not None and len(model.final_val_preds) > 0:
        all_preds_tensor = model.final_val_preds
        all_labels_tensor = model.final_val_labels
        all_thresholds_tensor = torch.tensor(model.final_thresholds)
        if all_preds_tensor.shape != all_labels_tensor.shape:    
            all_preds_tensor = all_preds_tensor.squeeze()
            all_labels_tensor = all_labels_tensor.squeeze()

        print("--- Generating ROC Curve ---")
        try:
            final_auc_score = TMF.auroc(all_preds_tensor, all_labels_tensor.int(), task="binary").item()
            print(f"Final Calculated AUC: {final_auc_score:.4f}")
            plot_roc_curve(predictions=all_preds_tensor, labels=all_labels_tensor, final_auc=final_auc_score, save_path=roc_plot_save_path)
        except Exception as e: print(f"Error plotting ROC curve: {e}")

        print("--- Generating ROC Curve For Error Maps---")
        try:
            final_auc_score_threshold = TMF.auroc(all_thresholds_tensor, all_labels_tensor.int(), task="binary").item()
            print(f"Final Calculated AUC for Error Maps: {final_auc_score_threshold:.4f}")
            plot_roc_curve(predictions=all_thresholds_tensor, labels=all_labels_tensor, final_auc=final_auc_score_threshold, save_path=roc_plot_save_path_threshold)
        except Exception as e: print(f"Error plotting ROC curve for error maps: {e}")

        print("--- Generating Confusion Matrix ---")
        try:
            pred_labels = (all_preds_tensor > args.threshold).int().numpy()
            true_labels = all_labels_tensor.int().numpy()
            plot_confusion_matrix(true_labels=true_labels, pred_labels=pred_labels, save_path=cm_save_path, class_names=['Normal', 'Anomaly'])
        except Exception as e: print(f"Error generating confusion matrix: {e}")

        print("--- Generating Error Map Confusion Matrix ---")
        try:
            threshold_pred_labels = (all_thresholds_tensor > args.cm_threshold).int().numpy()
            true_labels = all_labels_tensor.int().numpy()
            plot_confusion_matrix(true_labels=true_labels, pred_labels=threshold_pred_labels, save_path=cm_threshold_save_path, class_names=['Normal', 'Anomaly'])
        except Exception as e: print(f"Error generating confusion matrix for error maps: {e}")

    else:
        print("Skipping ROC/CM plots (no prediction data).")

    args_save_path = os.path.join(save_path, f'arguments_version_{version_number}.txt')
    try:
        with open(args_save_path, 'w') as f:
            args_dict = vars(args) 
            f.write(f"Arguments for Version {version_number}:\n")
            f.write("="*30 + "\n")
            for key, value in args_dict.items():
                f.write(f"{key}: {value}\n")
        print(f"Arguments saved successfully")
    except Exception as e:
        print(f"Error saving arguments to file: {e}")

    print(f"\n--- Script Finished (Mode: {args.mode}, Version: {version_number}) ---")
        
if __name__ == "__main__":
    main()
