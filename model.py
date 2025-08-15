# Cell 3: Neural Network Models
class AudioCNN(nn.Module):
    """CNN for processing individual audio clips (mel-spectrograms)"""
    
    def __init__(self, input_channels=1, embedding_dim=128):
        super(AudioCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25/2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25/2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25/2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(0.5/2)
        )
        
        # Calculate the flattened size
        self.flattened_size = 256 * 4 * 4
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5/2),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x

class DemographicMLP(nn.Module):
    """MLP for processing demographic features"""
    
    def __init__(self, input_dim=5, embedding_dim=128):
        super(DemographicMLP, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3/2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3/2),
            nn.Linear(128, embedding_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for combining location embeddings"""
    
    def __init__(self, embed_dim=128, num_heads=8, ff_dim=512, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        #self.layer_norm = nn.LayerNorm(embed_dim)
        # CHAT SUGG 2
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # ---- Multi-head attention sublayer ----
        # Pre-LN
        x_norm = self.layer_norm1(x)
        attn_output, attn_weights = self.multihead_attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Residual connection

        # ---- Feed-forward sublayer ----
        x_norm = self.layer_norm2(x)
        ff_output = self.ffn(x_norm)
        x = x + ff_output  # Residual connection

        return x, attn_weights