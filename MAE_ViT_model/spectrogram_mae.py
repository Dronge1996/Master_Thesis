import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl
from typing import Tuple
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class PatchEmbedding(pl.LightningModule):
    """
    Convert spectrogram to patch embeddings
    
    Args:
        img_size (tuple): Input spectrogram size (height, width)
        patch_size (tuple): Size of each patch (height, width)
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=(496, 496), patch_size=(16, 16), in_chans=1, embed_dim=256):
        super().__init__()
        
        # Calculate number of patches
        self.grid_size = (
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # Patch embedding layer
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size[0], 
            stride=patch_size[0]
        )
        
    def forward(self, x):
        """
        Convert input spectrogram to patch embeddings
        
        Args:
            x (torch.Tensor): Input spectrogram
        
        Returns:
            torch.Tensor: Patch embeddings
        """
        x = self.proj(x)  # Extract patches and project to embedding space
        x = x.flatten(2).transpose(1, 2)  # Flatten patches
        return x

class AsymmetricDecoder(pl.LightningModule):
    """
    Asymmetric decoder for MAE reconstruction
    
    Args:
        embed_dim (int): Input embedding dimension
        decoder_embed_dim (int): Decoder embedding dimension
        decoder_depth (int): Number of decoder layers
        decoder_num_heads (int): Number of attention heads in decoder
        patch_size (tuple): Size of input patches
        in_chans (int): Number of input channels
        num_patches (int): Total number of patches
    """
    def __init__(
        self, 
        embed_dim=256,  
        decoder_depth=4, 
        decoder_num_heads=4, 
        patch_size=(16, 16), 
        in_chans=1,
        num_patches=961
    ):
        super().__init__()
        # Mask token projection
        self.mask_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        
        # Add positional embedding for decoder
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Decoder embedding
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=decoder_num_heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.0
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, 
            num_layers=decoder_depth
        )
        
        # Final reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(embed_dim, patch_size[0] * patch_size[1] * in_chans),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(embed_dim)
        
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_patches = num_patches
        
    def forward(self, x, mask_indices, num_patches):
        """
        Forward pass for asymmetric decoder
        
        Args:
            x (torch.Tensor): Encoded visible patches
            mask_indices (torch.Tensor): Indices of masked patches
            num_patches (int): Total number of patches
        
        Returns:
            torch.Tensor: Reconstructed spectrogram patches
        """
        device = x.device
        # Project encoded patches to decoder dimension
        x = x.to(torch.float32)
        x = self.decoder_embed(x)
        mask_indices = mask_indices.to(device)
        
        # Prepare full sequence with mask tokens
        B = x.shape[0]
        full_sequence = torch.zeros(B, num_patches, x.shape[-1], device=device)
        
        # Insert visible patches into full sequence
        ones_tensor = torch.ones(num_patches, device=device)

        visible_indices = torch.where(
            ones_tensor.scatter(0, mask_indices[0], 0).bool()
        )[0]

        full_sequence[:, visible_indices] = x
        
        # Fill masked positions with learnable mask token
        mask_pos = torch.where(
            ones_tensor.scatter(0, visible_indices, 0).bool()
        )[0]
        full_sequence[:, mask_pos] = self.mask_token.to(device).expand(B, len(mask_pos), -1)
        
        # Add positional embeddings to the full sequence
        full_sequence = self.norm(full_sequence + self.decoder_pos_embed.to(device))

        # Decode full sequence
        decoded = self.decoder(full_sequence.transpose(0, 1)).transpose(0, 1)
        
        # Reconstruct patches
        reconstructed = self.reconstruction_head(decoded)
        reconstructed = reconstructed.reshape(
            B, num_patches, self.patch_size[0], self.patch_size[1]
        )
        
        return reconstructed

class MaskedAutoencoderViT(pl.LightningModule):
    """
    Vision Transformer Masked Autoencoder for Spectrogram Reconstruction
    
    Args:
        img_size (tuple): Input spectrogram size
        patch_size (tuple): Patch size for embedding
        in_chans (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer encoder layers
        num_heads (int): Number of attention heads
        mask_ratio (float): Proportion of patches to mask
    """
    def __init__(
        self, 
        img_size=(496, 496), 
        patch_size=(16, 16), 
        in_chans=1,
        embed_dim=256,  
        depth=8, 
        num_heads=8, 
        mask_ratio=0.75
    ):
        super().__init__()
        
        # Embedding dimension
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim
        )

        # Calculate number of patches
        self.num_patches = self.patch_embed.num_patches
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=self.embed_dim * 4, 
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=depth
        )
        
        # Asymmetric decoder
        self.decoder = AsymmetricDecoder(
            embed_dim=self.embed_dim,
            patch_size=patch_size,
            in_chans=in_chans,
            num_patches=self.num_patches
        )
        
        # Hyperparameters
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.pos_embed)
        
    def random_masking(self, x):
        """
        Randomly mask patches
        
        Args:
            x (torch.Tensor): Input patch embeddings
        
        Returns:
            torch.Tensor: Visible patch embeddings
            torch.Tensor: Mask indices
            torch.Tensor: Unmasked indices
        """
        device = x.device
        N, L, D = x.shape  # Batch, num_patches, embed_dim
        
        # Determine number of patches to mask
        num_mask = int(L * self.mask_ratio)
        
        # Generate random mask
        noise = torch.rand(N, L, device=device)
        shuffle_indices = torch.argsort(noise, dim=1)
        mask_indices = shuffle_indices[:, :num_mask]
        unmask_indices = shuffle_indices[:, num_mask:].to(device)

        # Select only visible patches
        visible_x = torch.gather(
            x, 
            dim=1, 
            index=unmask_indices.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )
        
        return visible_x, mask_indices, unmask_indices
    
    def forward(self, x):
        """
        Forward pass for MAE
        
        Args:
            x (torch.Tensor): Input spectrogram
        
        Returns:
            dict: Reconstruction and other relevant information
        """
        if isinstance(x, list):
            x = x[0]
        # Patch embedding   
        x_patch = self.patch_embed(x)
    
        # Add positional embedding
        x_patch += self.pos_embed
        
        # Random masking
        x_visible, mask_indices, unmask_indices = self.random_masking(x_patch)
        
        # Transformer encoder (only on visible patches)
        x_encoded = self.encoder(x_visible)
        
        # Decoder reconstruction
        x_reconstructed = self.decoder(x_encoded, mask_indices, self.num_patches)
        
        return {
            'reconstruction': x_reconstructed,
            'mask_indices': mask_indices,
            'unmask_indices': unmask_indices
        }

def extract_patches(x, patch_size=(16, 16)):
    """
    Extract patches from input tensor without embedding
    
    Args:
        x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
        patch_size (tuple): Size of each patch
        
    Returns:
        torch.Tensor: Patches of shape [batch_size, num_patches, patch_height, patch_width]
    """
    B, C, H, W = x.shape
    patch_h, patch_w = patch_size
    
    # Calculate number of patches
    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    
    # Reshape to extract patches
    patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
    patches = patches.contiguous().view(B, C, -1, patch_h, patch_w)
    patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_h, patch_w]
    
    if C == 1:
        patches = patches.squeeze(2)  # Remove channel dimension for single-channel input
        
    return patches

def compute_mae_loss(pred, target, mask_indices, patch_size=(16, 16)):
    """
    Compute Masked Autoencoder Loss
    
    Args:
        pred (torch.Tensor): Predicted reconstruction [B, num_patches, patch_h, patch_w]
        target (torch.Tensor): Original spectrogram [B, C, H, W]
        mask_indices (torch.Tensor): Indices of masked patches
        patch_size (tuple): Size of each patch
    
    Returns:
        torch.Tensor: Mean squared error loss
    """
    device = pred.device

    # Validate input ranges and check for NaN/Inf
    if torch.isnan(pred).any() or torch.isinf(pred).any():
        print("WARNING: NaN or Inf values detected in prediction tensor")
    
    # Convert targets if list
    if isinstance(target, list):
        target = target[0]
    
    if torch.isnan(target).any() or torch.isinf(target).any():
        print("WARNING: NaN or Inf values detected in target tensor")

    # Extract patches from target
    target_patches = extract_patches(target, patch_size)
    
    # Get batch size
    B = pred.shape[0]
    
    # Initialize tensor to hold losses for each batch item
    losses = []
    
    # Compute loss per batch item
    for b in range(B):
        # Get the masked indices for this batch item
        indices = mask_indices[b]
        
        # Get corresponding predictions and targets
        pred_patches = pred[b][indices]
        target_patches_b = torch.stack([target_patches[b][idx.item()] for idx in indices])

        # Compute MSE loss for this batch item
        batch_loss = F.mse_loss(pred_patches, target_patches_b)

        losses.append(batch_loss)


    
    # Average losses across batch
    return torch.stack(losses).mean()

def get_spectrogram_transforms(img_size: Tuple[int, int] = (496, 496)) -> transforms.Compose:
    """
    Create data transforms for spectrograms
    
    Args:
        img_size (tuple): Desired image size
    
    Returns:
        transforms.Compose: Composition of transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])

def main():
    # # Example usage
    # # Transforms
    # transform = get_spectrogram_transforms()
    
    # # Hyperparameters
    # batch_size = 16
    # img_size = (496, 496)
    # in_chans = 1
    
    # # Create random spectrogram
    # spectrograms = torch.randn(batch_size, in_chans, *img_size)
    # spectrograms = [spectrograms]
    
    # # Initialize MAE model
    # mae_model = MaskedAutoencoderViT(
    #     img_size=img_size, 
    #     patch_size=(16, 16), 
    #     in_chans=in_chans,
    #     embed_dim=256,
    #     mask_ratio=0.75
    # )
    
    # # Forward pass
    # output = mae_model(spectrograms)
    # # print(output['reconstruction'].shape)
    # # Compute loss
    # loss = compute_mae_loss(
    #     output['reconstruction'], 
    #     spectrograms, 
    #     output['mask_indices']
    # )
    
    # # print(f"Reconstruction Loss: {loss.item()}")
    pass

if __name__ == "__main__":
    main()