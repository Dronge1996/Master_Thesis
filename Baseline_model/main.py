import os
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset

from typing import Tuple
from tqdm import tqdm, trange

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, dropout = 0.2, input_shape = (1, 496, 496)):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # --- Calculate the flattened encoder size dynamically ---
        with torch.no_grad():
            # Create dummy input with expected size and batch size of 1
            dummy_input = torch.zeros(1, *self.input_shape)

            # Pass dummy input through encoder    
            encoder_output = self.encoder(dummy_input)

            # Define flattened encoder size
            encoder_output_features = encoder_output.shape[1] * encoder_output.shape[2] * encoder_output.shape[3]

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(encoder_output_features, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        flattened = encoded.view(encoded.size(0), -1)
        classified = self.classifier(flattened)
        return decoded, classified


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[index]

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

# Data loading
def create_dataloaders(data_dir, img_size=(496, 496), batch_size=32, num_workers=4, train_val_split=0.9):
    """Create train and validation dataloaders"""
    train_transform = get_spectrogram_transforms(img_size, type='train')
    val_transform = get_spectrogram_transforms(img_size, type='val')
    
    full_dataset = datasets.ImageFolder(
        root=data_dir, 
        transform=None
    )
    
    try:
        train_size = int(train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        if train_size == 0 or val_size == 0:
                raise ValueError("Split resulted in an empty dataset. Check data and split ratio.")
        train_subset_indices, val_subset_indices = torch.utils.data.random_split(
                range(len(full_dataset)), [train_size, val_size]
        )
    except ValueError as e:
        print(f"Error during dataset split: {e}")
        print(f"Full dataset size: {len(full_dataset)}, Train split ratio: {train_val_split}")
        # Handle error appropriately, maybe exit or return None
        return None, None
    
    train_subset = Subset(full_dataset, train_subset_indices.indices)
    val_subset = Subset(full_dataset, val_subset_indices.indices)

    # Wrap the subsets with the appropriate transforms
    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, val_loader

def get_spectrogram_transforms(img_size: Tuple[int, int] = (496, 496), type: str = 'train') -> transforms.Compose:
    """
    Create data transforms for spectrograms
    
    Args:
        img_size (tuple): Desired image size
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    if type == 'train':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
    elif type == 'val':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    elif type == 'common':
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    
    else:
        raise ValueError("Invalid type. Expected 'train', 'val' or 'common")

def train_model(model, train_loader, criterion_reconstruction, criterion_classification, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.unsqueeze(-1).float()
        
        optimizer.zero_grad()
        decoded, classified = model(images)

        loss_reconstruction = criterion_reconstruction(decoded, images)
        loss_classification = criterion_classification(classified, labels)
        loss = loss_reconstruction + loss_classification
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_model(model, 
                   val_loader, 
                   criterion_reconstruction, 
                   criterion_classification, 
                   device, 
                   threshold=0.5,
                   error_threshold=6200,
                   compute_metrics=False):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_probabilities = []
    all_sum_errors = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).float()

            decoded, classified = model(images)

            loss_reconstruction = criterion_reconstruction(decoded, images)
            loss_classification = criterion_classification(classified, labels)
            loss = loss_reconstruction + loss_classification
            val_loss += loss.item()

            probabilities = torch.sigmoid(classified)
            errors = torch.abs(images - decoded)
            sample_error_sum = []
            for i in range(images.shape[0]):
                sample_error_sum.append(torch.sum(errors[i]).item())
            
            if compute_metrics:
                # Store labels and probabilities for ROC/AUC and CM
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_sum_errors.extend(sample_error_sum)

            # Calculate accuracy using the threshold
            predictions = (probabilities > threshold).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)


    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    if compute_metrics:
        # Compute confusion matrix using the threshold
        binary_predictions = (np.array(all_probabilities) > threshold).astype(int)
        cm = confusion_matrix(all_labels, binary_predictions)

        threshold_predictions = (np.array(all_sum_errors) > error_threshold).astype(int)
        cm_error = confusion_matrix(all_labels, threshold_predictions)
        return val_loss, val_accuracy, all_labels, all_probabilities, all_sum_errors, cm, cm_error, decoded
    else:
        return val_loss, val_accuracy

def save_model(model, save_path, model_name="simple_ae.pth"):
    """
    Save the trained model to a specified path.

    Args:
        model (nn.Module): The model to save.
        save_path (str): The directory path where the model will be saved.
        model_name (str): The name of the model file.
    """
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, "models"))

    model_file = os.path.join(save_path, "models", model_name)
    torch.save(model.state_dict(), model_file)
    print(f"Model saved as {model_file}")

def plot_losses(train_losses, val_losses, plot_dir, version):

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    # plt.show()
    os.makedirs(os.path.join(plot_dir, "version_" + str(version)), exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "version_" + str(version), "losses.png"))

def plot_accuracy(val_accuracy, plot_dir, version):
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracy, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    # plt.show()
    os.makedirs(os.path.join(plot_dir, "version_" + str(version)), exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "version_" + str(version), "accuracy.png"))

def plot_confusion_matrix(cm, plot_dir, version, error=False):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    # plt.show()
    os.makedirs(os.path.join(plot_dir, "version_" + str(version)), exist_ok=True)
    if error:
        plt.savefig(os.path.join(plot_dir, "version_" + str(version), "confusion_matrix_error.png"))
    else:
        plt.savefig(os.path.join(plot_dir, "version_" + str(version), "confusion_matrix.png"))

def plot_auc_roc(y_true, y_scores, plot_dir, version, error=False):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.show()
    os.makedirs(os.path.join(plot_dir, "version_" + str(version)), exist_ok=True)
    if error:
        save_path = os.path.join(plot_dir, "version_" + str(version), "auc_roc_curve_error.png")
    else:
        save_path = os.path.join(plot_dir, "version_" + str(version), "auc_roc_curve.png")
    plt.savefig(save_path)
    print(f"AUC ROC curve saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Train a simple autoencoder for spectrogram classification")
    parser.add_argument('--batch_size',     type=int,   default=32,             help='Batch size for training')
    parser.add_argument('--num_epochs',     type=int,   default=200,            help='Number of epochs for training')
    parser.add_argument('--lr',             type=float, default=1e-5,           help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay',   type=float, default=1e-5,           help='Weight decay for the optimizer')
    parser.add_argument('--dropout',        type=float, default=0.2,            help='Dropout rate for linear classifier')
    parser.add_argument('--threshold',      type=float, default=0.5,            help='Threshold for binary classification')
    parser.add_argument('--cm_threshold',   type=float, default=6200,           help='Threshold for confusion matrix')
    parser.add_argument('--version',        type=int,   default=None,           help='Version number for the model')
    parser.add_argument('--model_name',     type=str,   default='simple_ae',    help='Name of the model')
    parser.add_argument('--data_dir',       type=str,   default=r'C:\Path\to\data',                      help='Directory containing the training data')
    parser.add_argument('--save_dir',       type=str,   default=r"C:\Path\to\where\models\are\saved",    help='Directory to save the model and results')
    parser.add_argument('--plot_dir',       type=str,   default=r"C:\Path\to\where\plots\are\saved",     help='Directory to save plots')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Determine version number ---
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
                        continue

        if not existing_versions:
            version_number = 0
        else:
            version_number = max(existing_versions) + 1
        print(f"--- Determined next available version: {version_number} ---")
    
    config = {
    'img_size': (496, 496),
    'batch_size': args.batch_size,
    'num_workers': 4,
    'num_epochs': args.num_epochs,
    'learning_rate': args.lr,
    'dropout': args.dropout,
    'weight_decay': args.weight_decay,
    'data_dir': args.data_dir,
    'save_dir': args.save_dir,
    'plot_dir': args.plot_dir,
    'version': version_number,
    'model_name': args.model_name,
    'threshold': args.threshold,
    'cm_threshold': args.cm_threshold,
    'pos_weight': 1,
}
    
    # Define the loss function and optimizer
    model = Autoencoder(dropout=config['dropout']).to(device)
    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config['pos_weight']).to(device))
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    train_loader, val_loader = create_dataloaders(
        config['data_dir'], 
        img_size=config['img_size'], 
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    best_model_wts = None
    
    # Training loop
    for epoch in trange(config['num_epochs'], desc="Epochs"):
        
        train_loss = train_model(model, train_loader, criterion_reconstruction, criterion_classification, optimizer, device)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion_reconstruction, criterion_classification, device, compute_metrics=False)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
         
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'Early stopping counter: {epochs_no_improve}/{patience}')

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    # Load the best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    print("Training completed.")

    # Saving the configuration as a txt file
    config_save_path = os.path.join(args.plot_dir, "version_" + str(version_number), f'config_version_{version_number}.txt')
    try:
        os.makedirs(os.path.dirname(config_save_path), exist_ok=True)
        with open(config_save_path, 'w') as f: 
            f.write(f"Configuration for Version {version_number}:\n")
            f.write("="*30 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        print(f"Configuration saved successfully")
    except Exception as e:
        print(f"Error saving configuration to file: {e}")
    
    # Save the trained model
    print(f'Saving model to {os.path.join(config["save_dir"], f"{config['model_name']}_{config['version']}.pth")}')
    save_model(model, save_path=config['save_dir'], model_name=f"{config['model_name']}_{config['version']}.pth")
    
    print("Plotting losses...")
    plot_losses(train_losses, val_losses, config['plot_dir'], config['version'])
    
    print("Plotting accuracy...")
    plot_accuracy(val_accuracies, config['plot_dir'], config['version'])
    
    # --- Generate final validation metrics and plots ---
    print("Calculating final validation metrics and plotting CM & AUC ROC...")
    # Call validate_model with compute_metrics=True
    _, _, final_labels, final_probabilities, final_sum_errors, cm, cm_error, _ = validate_model(
        model,
        val_loader,
        criterion_reconstruction,
        criterion_classification,
        device,
        compute_metrics=True,
        threshold=config['threshold'],
        error_threshold=config['cm_threshold']
    )

    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm, config['plot_dir'], config['version'])
    plot_confusion_matrix(cm_error, config['plot_dir'], config['version'], error=True)

    print("Plotting AUC ROC curve...")
    plot_auc_roc(final_labels, final_probabilities, config['plot_dir'], config['version'])
    

if __name__ == "__main__":
    main()
