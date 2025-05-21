import os
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback

class LossTracker(Callback):
    """
    PyTorch Lightning callback to track training and validation losses
    for visualization after training completes.
    
    This callback collects loss values after each training and validation epoch
    and provides a method to visualize the loss curves.
    """
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.current_epoch = 0
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Collect training loss at the end of each epoch"""
        # Get the smoothed training loss from the trainer
        train_loss = trainer.callback_metrics.get('train_loss_epoch')
        if train_loss is not None:
            # Add loss value to our list
            self.train_losses.append(train_loss.item())
            # Track the epoch number
            self.epochs.append(self.current_epoch)
            self.current_epoch += 1
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Collect validation loss at the end of each epoch"""
        # Get the validation loss from the trainer
        val_loss = trainer.callback_metrics.get('val_loss')
        if val_loss is not None:
            # Add loss value to our list
            self.val_losses.append(val_loss.item())
    
    def visualize_losses(self, save_path='./loss_curves.png'):
        """
        Visualize training and validation losses
        
        Args:
            save_path (str): Path to save the loss curve plot
        
        Returns:
            str: Path to the saved visualization
        """
        # Create figure and axis
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if self.train_losses:
            plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
            
        # Plot validation loss
        if self.val_losses:
            # Ensure validation epochs match training epochs
            # val_epochs = self.epochs[:len(self.val_losses)]

            plt.plot(self.epochs, self.val_losses[1:], 'r-', label='Validation Loss')
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MAE Training and Validation Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add text with final loss values
        if self.train_losses and self.val_losses:
            final_train = self.train_losses[-1]
            final_val = self.val_losses[-1]
            plt.text(0.02, 0.95, f'Final train loss: {final_train:.4f}\nFinal val loss: {final_val:.4f}', 
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss visualization saved to {save_path}")
        
        return save_path
    
def plot_final_learning_curves(trainer, save_path='./results/loss_curves.png'):
    """
    Alternative function that uses the PyTorch Lightning trainer's logged metrics
    to plot the learning curves after training.
    
    Args:
        trainer: PyTorch Lightning trainer instance after training
        save_path: Path to save the visualization
    
    Returns:
        str: Path to the saved visualization
    """
    # Extract metrics from trainer logs
    train_losses = []
    val_losses = []
    epochs = []
    
    # Get the logger
    logger = trainer.logger
    
    # Check if we have a TensorBoard logger
    if hasattr(logger, 'experiment'):
        # Try to get metrics from TensorBoard logger
        for event_accumulator in logger.experiment.event_accumulator:
            scalars = event_accumulator.scalars
            if 'train_loss_epoch' in scalars:
                train_losses = [s.value for s in scalars['train_loss_epoch']]
                epochs = [s.step for s in scalars['train_loss_epoch']]
            if 'val_loss' in scalars:
                val_losses = [s.value for s in scalars['val_loss']]
    
    # If we couldn't get metrics from logger, try from trainer
    if not train_losses or not val_losses:
        # Get metrics from trainer callback_metrics history if available
        if hasattr(trainer, 'callback_metrics'):
            for i, metrics in enumerate(trainer.callback_metrics):
                if 'train_loss_epoch' in metrics:
                    train_losses.append(metrics['train_loss_epoch'].item())
                    epochs.append(i)
                if 'val_loss' in metrics:
                    val_losses.append(metrics['val_loss'].item())
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    if train_losses:
        plt.plot(epochs if epochs else range(len(train_losses)), train_losses, 'b-', label='Training Loss')
        
    # Plot validation loss
    if val_losses:
        # Ensure validation epochs match training epochs
        val_epochs = epochs[:len(val_losses)] if epochs else range(len(val_losses))
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    
    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MAE Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add text with final loss values
    if train_losses and val_losses:
        final_train = train_losses[-1]
        final_val = val_losses[-1]
        plt.text(0.02, 0.95, f'Final train loss: {final_train:.4f}\nFinal val loss: {final_val:.4f}', 
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss visualization saved to {save_path}")
    
    return save_path

def plot_auc_curves(epochs, train_auc_history, val_auc_history, save_path='./auc_curves.png'):
    """
    Visualizes training and validation AUC curves.

    Args:
        epochs (list): List of epoch numbers.
        train_auc_history (list): List of training AUC values per epoch.
        val_auc_history (list): List of validation AUC values per epoch.
        save_path (str): Path to save the AUC curve plot.
    """
    plt.figure(figsize=(10, 6))

    # Ensure lengths match for plotting (validation might have one fewer entry initially depending on logging)
    min_len = min(len(epochs), len(train_auc_history), len(val_auc_history))
    epochs_plot = epochs[:min_len]
    train_auc_plot = train_auc_history[:min_len]
    val_auc_plot = val_auc_history[:min_len]


    if train_auc_plot:
        plt.plot(epochs_plot, train_auc_plot, 'b-', label='Training AUC')
    if val_auc_plot:
        plt.plot(epochs_plot, val_auc_plot, 'r-', label='Validation AUC')

    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC')
    # Set y-axis limits for AUC (typically 0 to 1)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add final values text
    if train_auc_plot and val_auc_plot:
        final_train_auc = train_auc_plot[-1]
        final_val_auc = val_auc_plot[-1]
        plt.text(0.02, 0.05, f'Final Train AUC: {final_train_auc:.4f}\nFinal Val AUC: {final_val_auc:.4f}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))


    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"AUC curve visualization saved to {save_path}")
    plt.close() # Close the figure