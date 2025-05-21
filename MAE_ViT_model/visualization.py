import os
import torch
import torchmetrics.functional as TMF
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def visualize_mae_results(model, dataloader, device='cuda', num_images=5, save_path='./mae_reconstructions.png'):
    """
    Visualize MAE model results with original, masked, and reconstructed spectrograms
    
    Args:
        model: Trained MAE model (MAELightningModule)
        dataloader: DataLoader containing validation data
        device: Device to run inference on
        num_images: Number of random images to visualize
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get a batch of images
    batch, _ = next(iter(dataloader))
    x = batch[0] if isinstance(batch, tuple) else batch

    # Select random images from batch
    batch_size = min(len(x), num_images)
    indices = torch.randperm(len(x))[:batch_size]
    images = [x[indices]]

    # Get original patch size from model
    patch_size = model.mae_model.patch_size
    img_size = [496]

    with torch.no_grad():
        # Get model output
        output = model.mae_model(images)
        
        reconstruction = output['reconstruction']
        mask_indices = output['mask_indices']
        
        # Create visualizations
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(batch_size):
            # Original image
            orig_img = images[0][i].cpu().squeeze().numpy()
            axes[i, 0].imshow(orig_img, cmap='viridis')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Create masked image
            masked_img = images[0][i].clone().cpu()
            
            # Create patch grid
            grid_h, grid_w = img_size[0] // patch_size[0], img_size[0] // patch_size[1]
            
            # Create mask
            mask = torch.ones((grid_h * grid_w), dtype=torch.bool)
            
            # Set masked patches to 0
            for idx in mask_indices[i]:
                mask[idx] = False
            
            mask = mask.reshape(grid_h, grid_w)
            
            # Apply mask by setting patches to gray (0.5)
            masked_img_np = masked_img.squeeze().numpy()
            for h in range(grid_h):
                for w in range(grid_w):
                    if not mask[h, w]:
                        h_start = h * patch_size[0]
                        h_end = h_start + patch_size[0]
                        w_start = w * patch_size[1]
                        w_end = w_start + patch_size[1]
                        masked_img_np[h_start:h_end, w_start:w_end] = 0.5
            
            axes[i, 1].imshow(masked_img_np, cmap='viridis')
            axes[i, 1].set_title('Masked Image')
            axes[i, 1].axis('off')
            
            # Reconstruct full image
            recon_img = masked_img.squeeze().clone().numpy()
            
            # Unflatten the reconstruction
            for idx in range(len(mask.flatten())):
                # Get 2D position from flattened index
                h, w = idx // grid_w, idx % grid_w
                h_start = h * patch_size[0]
                h_end = h_start + patch_size[0]
                w_start = w * patch_size[1]
                w_end = w_start + patch_size[1]
                
                # Replace masked region with reconstruction
                recon_patch = reconstruction[i, idx].cpu().numpy()
                recon_img[h_start:h_end, w_start:w_end] = recon_patch
            
            axes[i, 2].imshow(recon_img, cmap='viridis')
            axes[i, 2].set_title('Reconstructed Image')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
    return save_path

def visualize_anomaly_results(model,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_images: int = 5,
    save_path: str = './anomaly_visualization.png',
    threshold: float = 0.5
    ):
    """
    Visualizes results (Original, Error Map, Reconstruction) for the
    fine-tuned anomaly detection model.

    Args:
        model: The trained FineTuningLightningModule object.
        dataloader: DataLoader providing validation or test data (images, labels).
        device: Device to run inference on ('cuda' or 'cpu').
        num_images: Number of random images from the batch to visualize.
        save_path: Path to save the output visualization image.
        threshold: Probability threshold to determine predicted label (0 or 1).
    """
    # Ensure the passed model is on the correct device and in eval mode
    model.to(device)
    model.eval()
    model.freeze() # Ensure gradients are off and model is in eval mode
    print("Using provided trained model for visualization.")

    # Get a batch of images and labels
    try:
        batch = next(iter(dataloader))
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        
    except Exception as e:
        print(f"Error loading data batch: {e}")
        return

    # Select random images from the batch
    num_samples_in_batch = images.shape[0]
    if num_images > num_samples_in_batch:
        print(f"Warning: Requested {num_images} images, but batch only has {num_samples_in_batch}. Using {num_samples_in_batch}.")
        num_images = num_samples_in_batch

    indices = torch.randperm(num_samples_in_batch)[:num_images]
    images_to_show = images[indices]
    labels_to_show = labels[indices]

    # --- Get Model Outputs ---
    originals = []
    reconstructions = []
    error_maps = []
    predicted_logits = []

    with torch.no_grad():
        for i in range(num_images):
            img_single = images_to_show[i].unsqueeze(0) # Add batch dim

            # Perform inference using the model's components
            mae_output = model.mae_model(img_single)
            recon_patches = mae_output['reconstruction']
            recon_img = model.unpatchify(recon_patches) # Unpatchify to get full image
            error_map = torch.abs(img_single - recon_img) # Calculate error
            logits = model.classifier(error_map) # Get classifier output

            # Store results (move to CPU, remove batch dim)
            originals.append(img_single.squeeze(0).cpu())
            reconstructions.append(recon_img.squeeze(0).cpu())
            error_maps.append(error_map.squeeze(0).cpu())
            predicted_logits.append(logits.squeeze(0).cpu())

    # Stack results and calculate probabilities/labels
    predicted_logits_tensor = torch.stack(predicted_logits)
    probabilities = torch.sigmoid(predicted_logits_tensor).numpy()
    predicted_labels = (probabilities > threshold).astype(int)
    labels_to_show_np = labels_to_show.cpu().numpy()

    # --- Plotting ---
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images)) # 3 columns
    if num_images == 1:
         axes = axes.reshape(1, -1) # Adjust shape for single image case

    fig.suptitle("Fine-tuned Anomaly Detection Visualization", fontsize=16, y=1.02) # Adjust title pos

    for i in range(num_images):
        true_label = labels_to_show_np[i]
        pred_label = predicted_labels[i][0] # Classifier output is (1,)
        pred_score = probabilities[i][0]

        title_prefix = f"True: {true_label} | Pred: {pred_label} (Score: {pred_score:.2f})"

        # Column 1: Original image
        ax = axes[i, 0]
        img_np = originals[i].squeeze().numpy() # Remove channel dim if 1
        im = ax.imshow(img_np, cmap='viridis')
        ax.set_title(f"{title_prefix}\nOriginal Image")
        ax.axis('off')

        # Column 2: Error map
        ax = axes[i, 1]
        error_map_np = error_maps[i].squeeze().numpy()
        im_err = ax.imshow(error_map_np, cmap='hot')
        ax.set_title("Reconstruction Error Map")
        ax.axis('off')
        fig.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04)
        
        # Column 3: Reconstructed image
        ax = axes[i, 2]
        recon_img_np = reconstructions[i].squeeze().numpy()
        im_rec = ax.imshow(recon_img_np, cmap='viridis')
        ax.set_title("Reconstructed Image")
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Save the figure
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

def plot_roc_curve(predictions, labels, final_auc, save_path='./roc_curve.png'):
    """
    Computes ROC curve data and plots it.

    Args:
        predictions (torch.Tensor): Tensor of predicted probabilities (shape: [N,]).
        labels (torch.Tensor): Tensor of true labels (shape: [N,], dtype=torch.int).
        final_auc (float): The final calculated AUC score for the title.
        save_path (str): Path to save the ROC curve plot.
    """
    try:
        # Ensure labels are integer type
        labels = labels.int()
        # Compute ROC curve points using torchmetrics functional API
        fpr, tpr, thresholds = TMF.roc(predictions, labels, task="binary")

        plt.figure(figsize=(8, 8))
        plt.plot(fpr.numpy(), tpr.numpy(), color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {final_auc:.4f})') 
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.5)') # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)

        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve visualization saved to {save_path}")
        plt.close()

    except Exception as e:
        print(f"Error generating ROC curve: {e}")
        print("Ensure predictions and labels tensors are valid and have corresponding samples.")

def plot_error_maps_by_class(model,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_samples_per_class: int = 5,
    save_path: str = './error_map_comparison.png'
    ):
    """
    Finds samples of each class, calculates reconstruction error maps,
    and plots them for comparison.

    Args:
        model: The trained FineTuningLightningModule object.
        dataloader: DataLoader providing data (images, labels). Use training loader
                    to increase chances of finding enough anomaly samples.
        device: Device to run inference on ('cuda' or 'cpu').
        num_samples_per_class: Number of samples for each class (normal/anomaly) to plot.
        save_path: Path to save the output visualization image.
    """
    # Ensure model is on the correct device and in eval mode
    model.to(device)
    model.eval()
    model.freeze() # Ensure gradients are off

    print(f"Searching for {num_samples_per_class} samples of each class...")

    # Store collected data: {class_label: [(image, error_map), ...]}
    collected_samples = {0: [], 1: []}
    samples_found = {0: 0, 1: 0}
    max_batches_to_check = 100 # Limit search to avoid infinite loop if classes are missing
    batches_checked = 0

    # Iterate through dataloader to find samples
    with torch.no_grad():
        for batch in dataloader:
            if batches_checked >= max_batches_to_check:
                print("Warning: Reached max batch check limit before finding enough samples.")
                break
            batches_checked += 1

            images, labels = batch
            images = images.to(device)
            labels = labels.cpu().numpy() # Keep labels on CPU for indexing

            # Perform inference to get error maps
            mae_output = model.mae_model(images)
            recon_patches = mae_output['reconstruction']
            recon_images = model.unpatchify(recon_patches)
            error_maps = torch.abs(images - recon_images)

            # Check each sample in the batch
            for i in range(images.shape[0]):
                label = labels[i]
                if samples_found[label] < num_samples_per_class:
                    # Store original image and its error map (move to CPU)
                    collected_samples[label].append(
                        (images[i].cpu(), error_maps[i].cpu())
                    )
                    samples_found[label] += 1

            # Check if we found enough samples for both classes
            if samples_found[0] >= num_samples_per_class and samples_found[1] >= num_samples_per_class:
                print("Found enough samples for both classes.")
                break # Stop searching

    print(f"Samples collected: Normal={samples_found[0]}, Anomaly={samples_found[1]}")

    # --- Plotting ---
    num_rows = 2 # One row for normal, one for anomaly
    num_cols = num_samples_per_class
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3.5 * num_rows)) # Adjust size as needed

    fig.suptitle("Reconstruction Error Map Comparison (Normal vs. Anomaly)", fontsize=16, y=1.05)

    # Plot Normal Samples (Class 0) - Top Row
    for i in range(num_cols):
        ax = axes[0, i]
        if i < samples_found[0]:
            _, error_map_tensor = collected_samples[0][i]
            error_map_np = error_map_tensor.squeeze().numpy() # Remove channel if 1
            im = ax.imshow(error_map_np, cmap='hot')
            ax.set_title(f"Normal Sample {i+1}\nError Map")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Add colorbar
        else:
            ax.set_title(f"Normal Sample {i+1}\n(Not Found)")
            ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')

    # Plot Anomaly Samples (Class 1) - Bottom Row
    for i in range(num_cols):
        ax = axes[1, i]
        if i < samples_found[1]:
            _, error_map_tensor = collected_samples[1][i]
            error_map_np = error_map_tensor.squeeze().numpy() # Remove channel if 1
            im = ax.imshow(error_map_np, cmap='hot')
            ax.set_title(f"Anomaly Sample {i+1}\nError Map")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) # Add colorbar
        else:
            ax.set_title(f"Anomaly Sample {i+1}\n(Not Found)")
            ax.text(0.5, 0.5, 'N/A', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # Save the figure
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Error map comparison visualization saved to {save_path}")
    except Exception as e:
        print(f"Error saving error map comparison: {e}")
    finally:
        plt.close(fig) # Close the figure

def plot_error_distribution(errors_class_0, errors_class_1, save_path='./error_distribution.png', n_bins=50):
    """
    Plots the distribution of reconstruction error sums for two classes (e.g., normal vs. anomaly).

    Args:
        errors_class_0 (list): List of summed reconstruction errors for class 0 (normal).
        errors_class_1 (list): List of summed reconstruction errors for class 1 (anomaly).
        save_path (str): Path to save the distribution plot.
        n_bins (int): Number of bins for the histogram.
    """
    plt.figure(figsize=(10, 6))

    # Determine common range for bins
    all_errors = errors_class_0 + errors_class_1
    if not all_errors:
        print("Warning: No error data provided for distribution plot.")
        plt.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Reconstruction Error Distribution (No Data)')

    else:
        min_error = min(all_errors) if all_errors else 0
        max_error = max(all_errors) if all_errors else 1
        bin_edges = np.linspace(min_error, max_error, n_bins + 1)

        # Plot histograms with transparency
        if errors_class_0:
            plt.hist(errors_class_0, bins=bin_edges, alpha=0.6, label='Class 0 (Normal)', density=True, color='blue')
        if errors_class_1:
            plt.hist(errors_class_1, bins=bin_edges, alpha=0.6, label='Class 1 (Anomaly)', density=True, color='red')

        plt.xlabel('Summed Reconstruction Error per Sample')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors by Class')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Error distribution plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving error distribution plot: {e}")
    finally:
        plt.close()

def plot_confusion_matrix(true_labels, pred_labels, save_path='./confusion_matrix.png', class_names=None):
    """
    Plots a confusion matrix using seaborn heatmap.

    Args:
        true_labels (list or np.array): Ground truth labels.
        pred_labels (list or np.array): Predicted labels.
        save_path (str): Path to save the confusion matrix plot.
        class_names (list): List of class names for labeling the matrix.
    """
    try:
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Create a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        # Save the plot
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")

def single_image_full_visualize(model, dataloader, device='cuda', save_path='./results/single_reconstruction.png'):
    """
    Visualize MAE model results with original, masked, and reconstructed spectrograms
    
    Args:
        model: Trained MAE model (MAELightningModule)
        dataloader: DataLoader containing validation data
        device: Device to run inference on
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Get a batch of images
    batch, _ = next(iter(dataloader))
    x = batch[0] if isinstance(batch, tuple) else batch

    # Select a random imagesfrom batch
    batch_size = 1
    indices = torch.randperm(len(x))[:batch_size]
    images = [x[indices]]

    # Get original patch size from model
    patch_size = model.mae_model.patch_size
    img_size = [496]

    with torch.no_grad():
        # Get model output
        output = model.mae_model(images)
        
        reconstruction = output['reconstruction']
        mask_indices = output['mask_indices']
        
        # Create visualizations
        fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(batch_size):
            # Original image
            orig_img = images[0][i].cpu().squeeze().numpy()
            axes[i, 0].imshow(orig_img, cmap='viridis')
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Create masked image
            masked_img = images[0][i].clone().cpu()
            
            # Create patch grid
            grid_h, grid_w = img_size[0] // patch_size[0], img_size[0] // patch_size[1]
            
            # Create mask
            mask = torch.ones((grid_h * grid_w), dtype=torch.bool)
            
            # Set masked patches to 0
            for idx in mask_indices[i]:
                mask[idx] = False
            
            mask = mask.reshape(grid_h, grid_w)
            
            # Apply mask by setting patches to gray (0.5)
            masked_img_np = masked_img.squeeze().numpy()
            for h in range(grid_h):
                for w in range(grid_w):
                    if not mask[h, w]:
                        h_start = h * patch_size[0]
                        h_end = h_start + patch_size[0]
                        w_start = w * patch_size[1]
                        w_end = w_start + patch_size[1]
                        masked_img_np[h_start:h_end, w_start:w_end] = 0.5
            
            axes[i, 1].imshow(masked_img_np, cmap='viridis')
            axes[i, 1].set_title('Masked Image')
            axes[i, 1].axis('off')
            
            # Reconstruct full image all reconstructed patches
            recon_img = masked_img.squeeze().clone().numpy()
            
            # Unflatten the reconstruction
            for idx in range(len(mask.flatten())):
                # Get 2D position from flattened index
                h, w = idx // grid_w, idx % grid_w
                h_start = h * patch_size[0]
                h_end = h_start + patch_size[0]
                w_start = w * patch_size[1]
                w_end = w_start + patch_size[1]
                
                # Replace placeholders with reconstructed image
                recon_patch = reconstruction[i, idx].cpu().numpy()
                recon_img[h_start:h_end, w_start:w_end] = recon_patch
            
            axes[i, 2].imshow(recon_img, cmap='viridis')
            axes[i, 2].set_title('Reconstructed Image')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save the figure
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        plt.close(fig)  # Close the figure to free memory
        
    return save_path