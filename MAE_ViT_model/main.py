import os
import sys
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from spectrogram_mae import (MaskedAutoencoderViT, compute_mae_loss)
from visualization import visualize_mae_results, single_image_full_visualize, plot_roc_curve
from loss_visualization import LossTracker

class MAELightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Masked Autoencoder Training
    
    Args:
        lr (float): Learning rate
        weight_decay (float): Weight decay for optimizer
        img_size (tuple): Input image size
        patch_size (tuple): Patch size for embedding
        mask_ratio (float): Proportion of patches to mask
    """
    def __init__(
        self, 
        lr=1e-5, 
        weight_decay=1e-5,
        img_size=(496, 496),
        patch_size=(16, 16),
        mask_ratio=0.75
    ):
        super().__init__()
        self.patch_size = patch_size

        self.mae_model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio
        )

        # Save hyperparameters
        self.save_hyperparameters()

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """
        Called before a batch is transferred to the device.
        Ensures the model is on the correct device.
        """
        if hasattr(self, 'device') and self.device.type != 'cpu':
            self.to(self.device)
            # Also ensure the MAE model and its components are on the device
            self.mae_model.to(self.device)
        return batch


    def forward(self, x):
        """Forward pass through MAE model"""
        return self.mae_model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for MAE"""
        x = batch[0] if isinstance(batch, tuple) else batch

        output = self(x)
        loss = compute_mae_loss(
            output['reconstruction'], 
            x, 
            output['mask_indices'],
            patch_size=self.patch_size
        )
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for MAE"""
        x = batch[0] if isinstance(batch, tuple) else batch

        output = self(x)
        loss = compute_mae_loss(
            output['reconstruction'], 
            x, 
            output['mask_indices'],
            patch_size=self.patch_size
        )
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

def create_dataloaders(data_dir, img_size=(496, 496), batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

    dataset = datasets.ImageFolder(
        root=data_dir, 
        transform=transform
    )
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train MAE model for spectrograms')
    parser.add_argument('--version',        type=int,   default=None,   help='Version number for saving model and outputs')
    parser.add_argument('--lr',             type=float, default=1e-5,   help='Learning rate')
    parser.add_argument('--weight_decay',   type=float, default=1e-5,   help='Weight decay')
    parser.add_argument('--batch_size',     type=int,   default=32,     help='Batch size')
    parser.add_argument('--max_epochs',     type=int,   default=160,    help='Maximum number of epochs')
    parser.add_argument('--mask_ratio',     type=float, default=0.75,   help='Mask ratio')
    parser.add_argument('--patch_size',     type=int,   default=16,     help="Size of the patches used for the encoder")
    args = parser.parse_args()

    # Speed uptimization for CUDA Cores /// Reduces precision
    torch.set_float32_matmul_precision('high')    
    # Configuration
    data_dir = "C:/Users/112899/OneDrive - Grundfos/Documents/Kandidat project/Data/Spectrogram/Normal"
    results_dir = "C:/Users/112899/OneDrive - Grundfos/Documents/Kandidat project/Code/code_outputs"
    # Hyperparameters
    config = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'img_size': (496, 496),
        'patch_size': (args.patch_size, args.patch_size),
        'mask_ratio': args.mask_ratio,
        'version': args.version
    }

    version_number = None
    if args.version is not None:
        # Use the version provided by the user
        version_number = args.version
        print(f"--- Using specified save model version: {version_number} ---")
    else:
        # Automatically determine the next available version
        print(f"--- No save model version specified, determining next available version ---")
        os.makedirs(results_dir, exist_ok=True) # Ensure base directory exists
        existing_versions = []
        if os.path.exists(results_dir) and os.path.isdir(results_dir):
            for item in os.listdir(results_dir):
                if item.startswith("version_") and os.path.isdir(os.path.join(results_dir, item)):
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
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir, 
        img_size=config['img_size'], 
        batch_size=config['batch_size']
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='mae-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode='min'
    )

    loss_tracker = LossTracker()

    callbacks = [checkpoint_callback, early_stop_callback, loss_tracker]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir='./logs', 
        name='mae_spectrogram'
    )
    
    # Initialize Lightning Module
    mae_module = MAELightningModule(
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        mask_ratio=config['mask_ratio'],
    )
    mae_module = mae_module.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Trainer with explicit GPU configuration
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        callbacks=callbacks,
        logger=logger,
        precision='32',
        deterministic=True,
        enable_progress_bar=True
    )

    # Train the model
    trainer.fit(
        mae_module, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    # Save final model
    trainer.save_checkpoint(f'./models/final_mae_model_{version_number}.ckpt')

    # Create version directory if it doesn't exist
    os.makedirs(os.path.join(results_dir, f'version_{version_number}'), exist_ok=True)

    # Saving Model weights in smaller .pth file
    torch.save(mae_module.mae_model.state_dict(), os.path.join(results_dir, f'version_{version_number}', f'mae_model_{version_number}.pth'))

    # Save Training and Validation Loss Curves
    print("Generating loss curves...")
    loss_tracker.visualize_losses(save_path=os.path.join(results_dir, f'version_{version_number}', f'loss_curves_{version_number}.png'))

    # Add this visualization code here
    print("Generating final visualization...")
    visualize_mae_results(
        model=mae_module,
        dataloader=val_loader,
        device=mae_module.device,
        num_images=5,
        save_path=os.path.join(results_dir, f'version_{version_number}', f'group_mae_reconstructions_{version_number}.png')
    )
    single_image_full_visualize(
        model=mae_module,
        dataloader=val_loader,
        device=mae_module.device,
        save_path=os.path.join(results_dir, f'version_{version_number}', f'single_mae_reconstructions_{version_number}.png')
    )

    # Save config dictionary to a text file
    config_file_path = os.path.join(results_dir, f'version_{version_number}', f'config_{version_number}.txt')
    with open(config_file_path, 'w') as config_file:
        for key, value in config.items():
            config_file.write(f'{key}: {value}\n')
    print(f"Configuration saved to {config_file_path}")

if __name__ == "__main__":
    main()