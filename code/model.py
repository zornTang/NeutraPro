import torch
import torch.nn as nn
import torch.nn.functional as F

class NeutraPro(nn.Module):
    def __init__(self, feature_dim, max_length, n_heads, d_model, kernel_sizes=(3, 5, 7), dropout=0.5):
        """
        Initialize the Attention-CNN model for binary classification.

        Args:
            feature_dim (int): Input feature dimension (e.g., 1024).
            max_length (int): Maximum sequence length.
            n_heads (int): Number of attention heads.
            d_model (int): Dimension of the attention model.
            kernel_sizes (tuple): Kernel sizes for the CNN layers.
        """
        super(NeutraPro, self).__init__()
        self.feature_dim = feature_dim
        self.max_length = max_length
        self.n_heads = n_heads
        self.d_model = d_model

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, feature_dim))

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=n_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(feature_dim)

        # CNN Layers with multiple kernel sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
            for k in kernel_sizes
        ])
        
        # MLP for output, using three layers (two hidden layers + output layer)
        mlp_input_dim = len(kernel_sizes) * feature_dim
        hidden_dim1 = mlp_input_dim // 2
        hidden_dim2 = hidden_dim1 // 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x, mask=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, max_num_segments, max_length, feature_dim).
            mask (Tensor, optional): Mask tensor of shape (batch_size, max_num_segments, max_length).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1).
        """
        batch_size, max_num_segments, max_length, feature_dim = x.shape

        # Reshape to merge segments for attention: (batch_size * max_num_segments, max_length, feature_dim)
        x = x.view(batch_size * max_num_segments, max_length, feature_dim)
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values.")

        # Add positional encoding
        x = x + self.positional_encoding
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values after adding positional encoding.")

        # Prepare mask for attention: (batch_size * max_num_segments, max_length)
        if mask is not None:
            # Reshape mask for processing
            mask_view = mask.view(batch_size, max_num_segments, max_length)
            
            # Identify completely masked segments (all False)
            completely_masked = ~mask_view.any(dim=2)
            
            # For completely masked segments, set the first position to True
            # to avoid NaN in attention calculation
            for b in range(batch_size):
                for s in range(max_num_segments):
                    if completely_masked[b, s]:
                        mask_view[b, s, 0] = True
            
            # Reshape back for attention
            mask = mask_view.view(batch_size * max_num_segments, max_length)

        # Multi-Head Attention
        attn_output, _ = self.attention(x, x, x, key_padding_mask=~mask if mask is not None else None)
        if torch.isnan(attn_output).any():
            raise ValueError("Attention output contains NaN values.")
        
        x = self.attention_norm(x + attn_output)  # Residual connection + LayerNorm
        if torch.isnan(x).any():
            raise ValueError("Output tensor contains NaN values after attention and normalization.")

        # Reshape for CNN: (batch_size * max_num_segments, feature_dim, max_length)
        x = x.permute(0, 2, 1)
        if torch.isnan(x).any():
            raise ValueError("Input tensor contains NaN values before CNN layers.")

        # Apply CNN layers
        cnn_outputs = [conv(x) for conv in self.convs]  # List of tensors with shape (batch_size * max_num_segments, feature_dim, 1)
        cnn_outputs = torch.cat(cnn_outputs, dim=1)  # Concatenate along the channel dimension
        cnn_outputs = cnn_outputs.squeeze(-1)  # Remove the last dimension: (batch_size * max_num_segments, len(kernel_sizes) * feature_dim)

        # Reshape back to (batch_size, max_num_segments, -1)
        cnn_outputs = cnn_outputs.view(batch_size, max_num_segments, -1)
        if torch.isnan(cnn_outputs).any():
            raise ValueError("CNN output contains NaN values.")

        # Implement masked pooling if mask is provided
        if mask is not None:
            # Reshape mask to match segments
            mask_view = mask.view(batch_size, max_num_segments, max_length)
            
            # Calculate which segments are valid (have at least one True in mask)
            valid_segments = mask_view.any(dim=2).float().unsqueeze(-1)  # [batch_size, max_num_segments, 1]
            
            # Apply the segment mask to cnn_outputs
            masked_outputs = cnn_outputs * valid_segments
            
            # Count valid segments per sample (avoid division by zero)
            valid_count = valid_segments.sum(dim=1)
            valid_count = torch.clamp(valid_count, min=1.0)  # Ensure at least 1
            
            # Average only over valid segments
            pooled_output = masked_outputs.sum(dim=1) / valid_count
        else:
            # If no mask, use regular mean pooling
            pooled_output = cnn_outputs.mean(dim=1)
            
        if torch.isnan(pooled_output).any():
            raise ValueError("Pooled output contains NaN values.")

        # Fully connected layer
        output = self.mlp(pooled_output)
        if torch.isnan(output).any():
            raise ValueError("Final output contains NaN values.")

        return output