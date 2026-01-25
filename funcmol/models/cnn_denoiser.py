import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """
    3D Convolutional Block with GroupNorm, SiLU activation, optional Dropout, and residual connection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 n_groups=8, dropout=0.0, time_emb_dim=None, use_residual=True):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.use_residual = use_residual and (in_channels == out_channels)
        
        # Main convolution path
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = nn.SiLU()
        
        # Time embedding projection (if DDPM mode)
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        
        # Dropout
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
        # Shortcut connection if channels change
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, t_emb=None):
        """
        Args:
            x: [B, C, D, H, W] input tensor
            t_emb: [B, time_emb_dim] time embedding (optional)
        Returns:
            [B, C_out, D, H, W] output tensor
        """
        identity = x
        
        h = self.conv(x)
        h = self.norm(h)
        
        # Add time embedding if provided
        if t_emb is not None and self.time_emb_dim is not None:
            t_proj = self.time_mlp(t_emb)  # [B, C_out]
            # Reshape to [B, C_out, 1, 1, 1] for broadcasting
            t_proj = t_proj[:, :, None, None, None]
            h = h + t_proj
        
        h = self.act(h)
        h = self.dropout(h)
        
        # Add residual connection
        if self.use_residual:
            h = h + self.shortcut(identity)
        
        return h


class Down3DBlock(nn.Module):
    """
    Downsampling block with multiple Conv3D blocks (for depth) and max pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8, 
                 dropout=0.0, time_emb_dim=None, num_conv_blocks=2):
        super().__init__()
        
        # First conv changes channels (no residual since channels change)
        self.conv_blocks = nn.ModuleList([
            Conv3DBlock(in_channels, out_channels, kernel_size=kernel_size, 
                       padding=kernel_size//2, n_groups=n_groups, 
                       dropout=dropout, time_emb_dim=time_emb_dim,
                       use_residual=False)  # No residual when channels change
        ])
        
        # Additional conv blocks at same resolution (with residual connections)
        for _ in range(num_conv_blocks - 1):
            self.conv_blocks.append(
                Conv3DBlock(out_channels, out_channels, kernel_size=kernel_size, 
                           padding=kernel_size//2, n_groups=n_groups, 
                           dropout=dropout, time_emb_dim=time_emb_dim,
                           use_residual=True)  # Use residual when channels stay same
            )
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x, t_emb=None):
        """
        Args:
            x: [B, C_in, D, H, W]
            t_emb: [B, time_emb_dim] (optional)
        Returns:
            h: [B, C_out, D/2, H/2, W/2] downsampled output
            skip: [B, C_out, D, H, W] skip connection before pooling
        """
        h = x
        for conv_block in self.conv_blocks:
            h = conv_block(h, t_emb)
        
        skip = h
        h = self.pool(h)
        return h, skip


class Up3DBlock(nn.Module):
    """
    Upsampling block with transpose convolution, skip connection, and multiple Conv3D blocks.
    """
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=3, n_groups=8, 
                 dropout=0.0, time_emb_dim=None, num_conv_blocks=2):
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        
        # After concatenation with skip, channels will be out_channels + skip_channels
        self.conv_blocks = nn.ModuleList([
            Conv3DBlock(out_channels + skip_channels, out_channels, kernel_size=kernel_size, 
                       padding=kernel_size//2, n_groups=n_groups, 
                       dropout=dropout, time_emb_dim=time_emb_dim,
                       use_residual=False)  # No residual when channels change
        ])
        
        # Additional conv blocks at same resolution (with residual connections)
        for _ in range(num_conv_blocks - 1):
            self.conv_blocks.append(
                Conv3DBlock(out_channels, out_channels, kernel_size=kernel_size, 
                           padding=kernel_size//2, n_groups=n_groups, 
                           dropout=dropout, time_emb_dim=time_emb_dim,
                           use_residual=True)  # Use residual when channels stay same
            )
        
    def forward(self, x, skip, t_emb=None):
        """
        Args:
            x: [B, C_in, D, H, W] input from previous layer
            skip: [B, skip_channels, D*2, H*2, W*2] skip connection from encoder
            t_emb: [B, time_emb_dim] (optional)
        Returns:
            [B, C_out, D*2, H*2, W*2] upsampled output
        """
        h = self.upconv(x)
        
        # Handle size mismatch for odd grid sizes
        # If upsampled size doesn't match skip size, pad h to match
        if h.shape != skip.shape:
            # Calculate padding needed
            pad_d = skip.shape[2] - h.shape[2]
            pad_h = skip.shape[3] - h.shape[3]
            pad_w = skip.shape[4] - h.shape[4]
            
            # Pad h to match skip size (pad format: [left, right, top, bottom, front, back])
            h = torch.nn.functional.pad(h, [0, pad_w, 0, pad_h, 0, pad_d])
        
        # Concatenate with skip connection
        h = torch.cat([h, skip], dim=1)
        
        # Apply multiple conv blocks for depth
        for conv_block in self.conv_blocks:
            h = conv_block(h, t_emb)
        
        return h


class CNNDenoiser(nn.Module):
    """
    3D CNN-based denoiser with U-Net architecture for molecular code denoising.
    
    Args:
        code_dim: Dimension of input code (e.g., 384)
        hidden_channels: List of channel sizes for each layer (e.g., [64, 128, 256])
        num_layers: Number of encoder/decoder layers
        kernel_size: Convolution kernel size
        dropout: Dropout rate
        grid_size: Size of 3D grid (e.g., 8 for 8x8x8)
        time_emb_dim: Dimension of time embedding for DDPM (None for non-DDPM mode)
    """
    def __init__(self, code_dim=384, hidden_channels=None, num_layers=3, 
                 kernel_size=3, dropout=0.1, grid_size=8, time_emb_dim=None,
                 num_conv_blocks_per_layer=2):
        super().__init__()
        
        self.code_dim = code_dim
        self.grid_size = grid_size
        self.num_layers = num_layers
        self.time_emb_dim = time_emb_dim
        self.num_conv_blocks_per_layer = num_conv_blocks_per_layer
        
        # Default hidden channels if not provided
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        
        # Ensure hidden_channels matches num_layers
        if len(hidden_channels) < num_layers:
            # Extend the list by repeating the last element
            hidden_channels = hidden_channels + [hidden_channels[-1]] * (num_layers - len(hidden_channels))
        elif len(hidden_channels) > num_layers:
            # Truncate the list
            hidden_channels = hidden_channels[:num_layers]
        
        self.hidden_channels = hidden_channels
        
        # Calculate number of groups for GroupNorm (must divide all channel sizes)
        self.n_groups = 8
        
        # Input projection: code_dim -> hidden_channels[0]
        self.input_proj = nn.Conv3d(code_dim, hidden_channels[0], kernel_size=1)
        
        # Time embedding projection (if DDPM mode)
        if time_emb_dim is not None:
            self.time_projection = nn.Linear(time_emb_dim, time_emb_dim)
        
        # Encoder blocks
        self.down_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels[i] if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            self.down_blocks.append(
                Down3DBlock(in_ch, out_ch, kernel_size=kernel_size, 
                           n_groups=self.n_groups, dropout=dropout, 
                           time_emb_dim=time_emb_dim,
                           num_conv_blocks=num_conv_blocks_per_layer)
            )
        
        # Bottleneck - also use multiple blocks for depth
        bottleneck_ch = hidden_channels[-1]
        bottleneck_blocks = []
        # First block expands channels (no residual)
        bottleneck_blocks.append(
            Conv3DBlock(bottleneck_ch, bottleneck_ch * 2, kernel_size=kernel_size, 
                       padding=kernel_size//2, n_groups=self.n_groups, 
                       dropout=dropout, time_emb_dim=time_emb_dim,
                       use_residual=False)
        )
        # Additional blocks at expanded dimension (with residual)
        for _ in range(num_conv_blocks_per_layer - 1):
            bottleneck_blocks.append(
                Conv3DBlock(bottleneck_ch * 2, bottleneck_ch * 2, kernel_size=kernel_size, 
                           padding=kernel_size//2, n_groups=self.n_groups, 
                           dropout=dropout, time_emb_dim=time_emb_dim,
                           use_residual=True)
            )
        # Final block contracts channels (no residual)
        bottleneck_blocks.append(
            Conv3DBlock(bottleneck_ch * 2, bottleneck_ch, kernel_size=kernel_size, 
                       padding=kernel_size//2, n_groups=self.n_groups, 
                       dropout=dropout, time_emb_dim=time_emb_dim,
                       use_residual=False)
        )
        self.bottleneck = nn.ModuleList(bottleneck_blocks)
        
        # Decoder blocks (reverse order)
        self.up_blocks = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = hidden_channels[i]
            skip_ch = hidden_channels[i]  # Skip connection comes from the same encoder layer
            out_ch = hidden_channels[i-1] if i > 0 else hidden_channels[0]
            self.up_blocks.append(
                Up3DBlock(in_ch, skip_ch, out_ch, kernel_size=kernel_size, 
                         n_groups=self.n_groups, dropout=dropout, 
                         time_emb_dim=time_emb_dim,
                         num_conv_blocks=num_conv_blocks_per_layer)
            )
        
        # Output projection: hidden_channels[0] -> code_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(self.n_groups, hidden_channels[0]),
            nn.SiLU(),
            nn.Conv3d(hidden_channels[0], code_dim, kernel_size=1)
        )
        
        # Initialize output layer with small values (better than zeros for deep networks)
        # Use small random initialization instead of zeros to help gradient flow
        nn.init.normal_(self.output_proj[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_proj[-1].bias)
    
    def forward(self, y, t=None):
        """
        Forward pass through the CNN denoiser.
        
        Args:
            y: Input tensor [B, n_grid, code_dim] or [B, 1, n_grid, code_dim]
            t: Time steps [B] for DDPM (optional)
        
        Returns:
            Output tensor [B, n_grid, code_dim]
        """
        # Handle input dimensions
        if y.dim() == 4:  # [B, 1, n_grid, code_dim]
            y = y.squeeze(1)
        elif y.dim() != 3:
            raise ValueError(f"Expected 3D or 4D input, got {y.dim()}D: {y.shape}")
        
        batch_size, n_grid, code_dim = y.shape
        
        # Verify grid size
        expected_n_grid = self.grid_size ** 3
        if n_grid != expected_n_grid:
            raise ValueError(
                f"Input grid size mismatch: expected n_grid={expected_n_grid} "
                f"(grid_size={self.grid_size}^3), but got n_grid={n_grid}"
            )
        
        # Reshape to 3D grid: [B, n_grid, code_dim] -> [B, code_dim, D, H, W]
        y = y.view(batch_size, self.grid_size, self.grid_size, self.grid_size, code_dim)
        y = y.permute(0, 4, 1, 2, 3)  # [B, code_dim, D, H, W]
        
        # Process time embedding if provided
        t_emb = None
        if t is not None and self.time_emb_dim is not None:
            from funcmol.models.ddpm import get_time_embedding
            t_emb = get_time_embedding(t, self.time_emb_dim)  # [B, time_emb_dim]
            t_emb = self.time_projection(t_emb)  # [B, time_emb_dim]
        
        # Input projection
        h = self.input_proj(y)  # [B, hidden_channels[0], D, H, W]
        
        # Encoder with skip connections
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)
        
        # Bottleneck - apply all bottleneck blocks
        for bottleneck_block in self.bottleneck:
            h = bottleneck_block(h, t_emb)
        
        # Decoder with skip connections
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)
        
        # Output projection
        output = self.output_proj(h)  # [B, code_dim, D, H, W]
        
        # Reshape back to [B, n_grid, code_dim]
        output = output.permute(0, 2, 3, 4, 1)  # [B, D, H, W, code_dim]
        output = output.reshape(batch_size, n_grid, code_dim)
        
        return output
