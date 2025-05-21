# test_simple_ae_imported.py

import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix # Keep for final CM calculation
import matplotlib.pyplot as plt # Keep for general plotting setup if needed
import torch
import torch.nn as nn # Keep for loss functions
import torchvision.datasets as datasets # Need for test loader
import torchvision.transforms as transforms # Need for test loader
from torch.utils.data import DataLoader # Need for test loader
from typing import Tuple

# --- Import necessary components from main.py ---
try:
    from main import (
        Autoencoder,
        validate_model,
        plot_confusion_matrix,
        plot_auc_roc
    )
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    print("Please ensure 'main.py' is in the same directory or accessible in the Python path,")
    print("and that it defines Autoencoder, validate_model, plot_confusion_matrix, plot_auc_roc.")
    exit()

# --- Helper Functions Specific to Testing (Data Loading) ---

def get_test_transforms(img_size: Tuple[int, int] = (496, 496)) -> transforms.Compose:
    """Get transforms for testing (no augmentation)."""
    # Define test transforms - should match preprocessing used during training, minus augmentation
    # This is adapted from the transforms defined inside create_dataloaders in main.py
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

def create_test_loader(data_dir, img_size=(496, 496), batch_size=32, num_workers=4, loader_type='val'):
    """Creates a DataLoader for testing (loads all data from dir)."""
    test_transform = get_test_transforms(img_size)
    try:
        test_dataset = datasets.ImageFolder(
            root=data_dir,
            transform=test_transform
        )
        if not test_dataset:
            print(f"Error: No images found in {data_dir} or its subdirectories.")
            return None
    except FileNotFoundError:
        print(f"Error: Data directory not found: {data_dir}")
        return None
    except Exception as e:
        print(f"An error occurred loading the dataset: {e}")
        return None


    print(f"Found {len(test_dataset)} testing samples in {data_dir}")
    # Split the dataset into train and validation sets (90/10 split)
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.1 * dataset_size))
    np.random.shuffle(indices)
    val_indices, train_indices = indices[:split], indices[split:]

    train_subset = torch.utils.data.Subset(test_dataset, train_indices)
    val_subset = torch.utils.data.Subset(test_dataset, val_indices)

    if loader_type == 'train':
        return DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    elif loader_type == 'val':
        return DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    elif loader_type == 'all':
    # Optionally, you can return both loaders and let the user choose
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        print(f"Error: Invalid loader_type '{loader_type}'. Use 'train', 'val', or 'all'.")
        return None

# --- Main Testing Function ---
def test_main():
    parser = argparse.ArgumentParser(description="Test a trained Simple Autoencoder model (imports from main.py)")
    parser.add_argument('--model_path',     type=str,   default="C:/Path/to/trained/baseline/model",  help='Path to the saved .pth model state_dict file')
    parser.add_argument('--data_dir',       type=str,   default=r'C:\Path\to\data',  help='Directory containing the validation/test data (ImageFolder structure)')
    parser.add_argument('--results_dir',    type=str,   default=r"C:\Path\to\where\results\are\saved",  help='Directory to save the output plots')
    parser.add_argument('--plot_dir',       type=str,   default=r"C:\Path\to\where\plots\are\saved", help='Directory to save plots')
    parser.add_argument('--version',        type=str,   default=None,   help='Version identifier for saving plot files (e.g., the model version)')
    parser.add_argument('--lmversion',      type=int,   default=4,      help='Load model from specific trained simple model')
    parser.add_argument('--batch_size',     type=int,   default=32,     help='Batch size for evaluation')
    parser.add_argument('--img_height',     type=int,   default=496,    help='Image height')
    parser.add_argument('--img_width',      type=int,   default=496,    help='Image width')
    parser.add_argument('--threshold',      type=float, default=0.5,    help='Decision threshold for confusion matrix calculation')
    parser.add_argument('--cm_threshold',   type=float, default=6200,   help='Threshold for confusion matrix calculation')
    parser.add_argument('--num_workers',    type=int,   default=0,      help='Number of dataloader workers')
    parser.add_argument('--dropout',        type=float, default=0.2,    help='Dropout probability used IN THE LOADED MODEL (must match training)')
    parser.add_argument('--loader_type',    type=str,   default='val',   help='Type of DataLoader to create (train, val, all)')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_save_dir = args.plot_dir
    version_number = None

    if args.version is not None:
        # Use the version provided by the user
        version_number = args.version
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
        
    # --- 1. Instantiate Model ---
    img_size = (args.img_height, args.img_width)
    input_channels = 1
    model_input_shape = (input_channels, img_size[0], img_size[1])

    # Instantiate the imported Autoencoder model
    # Make sure the arguments match the class definition in main.py
    # Assuming it takes dropout_prob and input_shape if you used the dynamic version
    try:
        model = Autoencoder(dropout=args.dropout, input_shape=model_input_shape)
    except TypeError:
         print("Trying to initialize Autoencoder without dropout/input_shape arguments...")
         try:
            # Fallback to original definition if dynamic one wasn't used in main.py
            model = Autoencoder()
         except Exception as e:
             print(f"Failed to initialize Autoencoder model. Error: {e}")
             print("Ensure the Autoencoder class is defined correctly in main.py.")
             return


    # --- 2. Load Saved Weights ---
    true_model_path = args.model_path.format(args.lmversion)
    if os.path.exists(true_model_path):
        try:
            model.load_state_dict(torch.load(true_model_path, map_location=device))
            print(f"Successfully loaded model weights from {true_model_path}")
        except Exception as e:
            print(f"Error loading state_dict: {e}")
            print("Ensure the model architecture definition in main.py matches the saved weights.")
            return
    else:
        print(f"Error: Model path not found: {true_model_path}")
        return

    # --- 3. Prepare for Evaluation ---
    model.to(device)
    model.eval() # Set model to evaluation mode

    # --- 4. Load Data ---
    test_loader = create_test_loader(
        args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loader_type=args.loader_type
    )
    if test_loader is None:
        print("Failed to create test data loader. Exiting.")
        return

    # Define criteria (needed for imported validate_model function signature)
    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.BCEWithLogitsLoss()

    # --- 5. Run Evaluation (using imported function) ---
    print(f"\nRunning evaluation on data from: {args.data_dir}")
    # Call imported validate_model to get outputs. Pass compute_metrics=True.
    try:
        val_loss, val_accuracy, final_labels, final_probabilities, final_error_sum, cm, cm_error, decoded_image = validate_model(
            model,
            test_loader,
            criterion_reconstruction,
            criterion_classification,
            device,
            threshold=args.threshold, 
            error_threshold=args.cm_threshold, 
            compute_metrics=True
        )
    except Exception as e:
        print(f"An error occurred during model validation: {e}")
        return


    if final_labels is None or final_probabilities is None:
        print("Error: Evaluation did not return labels or probabilities.")
        return

    print(f"\nEvaluation Results:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy (at threshold {args.threshold}): {val_accuracy:.2f}%")

    # # Convert results to numpy arrays
    # true_labels_np = np.array(final_labels).astype(int)
    # probabilities_np = np.array(final_probabilities)

    # if len(true_labels_np) == 0:
    #     print("Error: No samples processed during evaluation.")
    #     return

    # # --- 6. Apply Threshold and Calculate Final CM ---
    # final_decision_threshold = args.threshold
    # print(f"Applying final decision threshold for Confusion Matrix: {final_decision_threshold}")
    # predicted_labels_np = (probabilities_np > final_decision_threshold).astype(int)

    # # Use sklearn's confusion_matrix directly here for clarity
    # final_cm = confusion_matrix(true_labels_np, predicted_labels_np)
    # print(f"Confusion Matrix (Threshold={final_decision_threshold}):\n{final_cm}")

    # --- 7. Generate Plots (using imported functions) ---
    print("\nGenerating plots...")
    results_save_dir = os.path.join(args.results_dir, f"version_{args.version}")
    os.makedirs(results_save_dir, exist_ok=True)

    # Plot Confusion Matrix
    # Assuming plot_confusion_matrix from main.py takes cm, plot_dir, version_tag
    try:
        plot_confusion_matrix(cm, args.plot_dir, args.version)
        plot_confusion_matrix(cm_error, args.plot_dir, args.version, error=True)
        # Note: The original plot_confusion_matrix saved inside version_{version}. Adjust path if needed.
        # Example adjusted call if it needs the full save path:
        # cm_save_path = os.path.join(results_save_dir, f"test_confusion_matrix_thresh{args.threshold}.png")
        # plot_confusion_matrix(final_cm, cm_save_path) # If it takes full path
    except Exception as e:
        print(f"Error calling plot_confusion_matrix: {e}")


    # Plot AUC ROC Curve
    # Assuming plot_auc_roc from main.py takes true_labels, probabilities, plot_dir, version_tag
    try:
        plot_auc_roc(final_labels, final_probabilities, args.plot_dir, args.version)
        plot_auc_roc(final_labels, final_error_sum, args.plot_dir, args.version, error=True)
        # Note: The original plot_auc_roc saved inside version_{version}. Adjust path if needed.
        # Example adjusted call if it needs the full save path:
        # roc_save_path = os.path.join(results_save_dir, "test_auc_roc_curve.png")
        # plot_auc_roc(true_labels_np, probabilities_np, roc_save_path) # If it takes full path
    except Exception as e:
        print(f"Error calling plot_auc_roc: {e}")

    # Plot decoded image
    try:
        # Assuming decoded_image is a tensor, convert it to numpy for plotting
        decoded_image_np = decoded_image.cpu().detach().numpy()
        # If the decoded image is a batch, take the first image
        if len(decoded_image_np.shape) == 4:  # (batch_size, channels, height, width)
            decoded_image_np = decoded_image_np[0]
        # Remove channel dimension if it's single-channel
        if decoded_image_np.shape[0] == 1:
            decoded_image_np = decoded_image_np[0]

        # Plot the decoded image
        plt.figure(figsize=(6, 6))
        plt.imshow(decoded_image_np, cmap='viridis')
        plt.axis('off')
        plt.tight_layout()
        decoded_image_save_path = os.path.join(results_save_dir, "decoded_image.png")
        plt.savefig(decoded_image_save_path, bbox_inches='tight')
        plt.close()
        print(f"Decoded image saved at: {decoded_image_save_path}")
    except Exception as e:
        print(f"Error plotting or saving decoded image: {e}")

    print(f"\nTesting finished. Plots saved in directory: {results_save_dir}")

if __name__ == "__main__":
    test_main()
