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

class HeartSoundClassifier(nn.Module):
    """Complete heart sound classification model"""
    
    def __init__(self, embedding_dim=128, num_classes=2):
        super(HeartSoundClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.locations = ['AV', 'PV', 'TV', 'MV']
        
        # CNN for processing audio clips
        self.audio_cnn = AudioCNN(embedding_dim=embedding_dim)
        
        # MLP for processing demographics
        self.demographic_mlp = DemographicMLP(embedding_dim=embedding_dim)
        
        # Attention mechanism
        self.attention = MultiHeadAttention(embed_dim=embedding_dim)
        
        # Final classification layers
        # 5 embeddings (4 locations + demographics) * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(5 * embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5/2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5/2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, location_data, demographics):
        batch_size = demographics.size(0)
        location_embeddings = []
        
        # Process each location
        for location in self.locations:
            location_clips = location_data[location]  # Shape: (batch_size, num_clips, channels, height, width)
            
            # Process all clips for this location in the batch
            batch_location_embeddings = []
            
            for b in range(batch_size):
                clips = location_clips[b]  # Shape: (num_clips, channels, height, width)
                
                if clips.size(0) > 0:
                    # Process all clips through CNN
                    clip_embeddings = self.audio_cnn(clips)  # Shape: (num_clips, embedding_dim)
                    # Take mean across clips to get location embedding
                    location_embedding = clip_embeddings.mean(dim=0)  # Shape: (embedding_dim,) #YOSSI CHANGE
                    #location_embedding, _ = torch.max(clip_embeddings, dim=0) #CHAT SUGGESTIONS
                    #print(f"first 5 {clip_embeddings[:,0:5]}, mean: {location_embedding[0:5]}")
                    #location_embedding = clip_embeddings[0]  # Shape: (embedding_dim,)
                else:
                    # Handle case where no clips are available
                    location_embedding = torch.zeros(self.embedding_dim, device=demographics.device)
                
                batch_location_embeddings.append(location_embedding)
            
            # Stack embeddings for all samples in batch
            batch_location_tensor = torch.stack(batch_location_embeddings)  # Shape: (batch_size, embedding_dim)
            location_embeddings.append(batch_location_tensor)
        
        # Process demographics
        demographic_embeddings = self.demographic_mlp(demographics)  # Shape: (batch_size, embedding_dim)
        
        # Combine all embeddings
        all_embeddings = location_embeddings + [demographic_embeddings]
        combined_embeddings = torch.stack(all_embeddings, dim=1)  # Shape: (batch_size, 5, embedding_dim)
        
        # Apply attention
        attended_embeddings, attention_weights = self.attention(combined_embeddings)
        
        # Flatten for final classification
        flattened = attended_embeddings.view(batch_size, -1)  # Shape: (batch_size, 5 * embedding_dim)
        
        # Final classification
        output = self.classifier(flattened)
        
        return output
