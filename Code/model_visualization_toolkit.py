# Cell 6: Model Visualization Toolkit
# Comprehensive visualization for heart sound classification model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.hooks import RemovableHandle
import cv2
from typing import Dict, List, Tuple, Optional

class ModelVisualizer:
    """Complete visualization toolkit for heart sound model"""
    
    def __init__(self, model, device, val_loader=None, test_loader=None):
        self.model = model
        self.device = device
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Storage for activations and gradients
        self.activations = {}
        self.gradients = {}
        self.hooks = []
        
        # Cache for frequently used data
        self.cached_samples = None
        self.cached_labels = None
        self.cached_predictions = None
        
        print(f"ModelVisualizer initialized with device: {device}")
        if val_loader:
            print(f"Validation loader: {len(val_loader)} batches")
        if test_loader:
            print(f"Test loader: {len(test_loader)} batches")
    
    def cache_samples(self, num_samples=20, use_test=False):
        """Cache samples for quick visualization without reloading"""
        loader = self.test_loader if use_test else self.val_loader
        if loader is None:
            print("No data loader available!")
            return
            
        print(f"Caching {num_samples} samples...")
        
        all_location_data = {loc: [] for loc in ['AV', 'PV', 'TV', 'MV']}
        all_demographics = []
        all_labels = []
        all_predictions = []
        
        self.model.eval()
        samples_collected = 0
        
        with torch.no_grad():
            for location_data, demographics, labels in loader:
                if samples_collected >= num_samples:
                    break
                    
                # Move to device
                demographics = demographics.to(self.device)
                labels = labels.to(self.device)
                
                for loc in location_data:
                    for i in range(len(location_data[loc])):
                        location_data[loc][i] = location_data[loc][i].to(self.device)
                
                # Get predictions
                outputs = self.model(location_data, demographics)
                _, predicted = torch.max(outputs.data, 1)
                
                # Store data
                batch_size = min(labels.size(0), num_samples - samples_collected)
                for i in range(batch_size):
                    # Store location data
                    for loc in ['AV', 'PV', 'TV', 'MV']:
                        all_location_data[loc].append(location_data[loc][i])
                    
                    all_demographics.append(demographics[i])
                    all_labels.append(labels[i].item())
                    all_predictions.append(predicted[i].item())
                    
                    samples_collected += 1
                    if samples_collected >= num_samples:
                        break
        
        self.cached_samples = all_location_data
        self.cached_demographics = torch.stack(all_demographics)
        self.cached_labels = all_labels
        self.cached_predictions = all_predictions
        
        print(f"Cached {len(self.cached_labels)} samples")
        print(f"Normal: {sum(1 for l in all_labels if l == 0)}, Abnormal: {sum(1 for l in all_labels if l == 1)}")
    
    def register_hooks(self):
        """Register hooks to capture activations and gradients"""
        self.clear_hooks()
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(model, grad_input, grad_output):
                if isinstance(grad_output[0], torch.Tensor):
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for key layers
        layers_to_hook = {
            'cnn_conv1': self.model.audio_cnn.conv_layers[0],  # First conv layer
            'cnn_conv2': self.model.audio_cnn.conv_layers[4],  # Second conv layer  
            'cnn_conv3': self.model.audio_cnn.conv_layers[8],  # Third conv layer
            'cnn_conv4': self.model.audio_cnn.conv_layers[12], # Fourth conv layer
            'cnn_fc1': self.model.audio_cnn.fc_layers[0],      # First FC layer
            'cnn_output': self.model.audio_cnn.fc_layers[-1],  # CNN output
            'demo_mlp': self.model.demographic_mlp.layers[-1], # Demographics MLP output
            'attention': self.model.attention.multihead_attn,   # Attention layer
            'classifier': self.model.classifier[-1]            # Final classifier
        }
        
        for name, layer in layers_to_hook.items():
            handle = layer.register_forward_hook(get_activation(name))
            self.hooks.append(handle)
            handle = layer.register_backward_hook(get_gradient(name))
            self.hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def visualize_attention_weights(self, sample_indices=None, num_samples=6):
        """Visualize attention weights for cached samples"""
        if self.cached_samples is None:
            print("No cached samples! Call cache_samples() first.")
            return
        
        if sample_indices is None:
            sample_indices = list(range(min(num_samples, len(self.cached_labels))))
        
        # Modify model to return attention weights
        original_forward = self.model.forward
        attention_weights_storage = []
        
        def forward_with_attention(location_data, demographics):
            batch_size = demographics.size(0)
            location_embeddings = []
            
            for location in self.model.locations:
                location_clips = location_data[location]
                batch_location_embeddings = []
                
                for b in range(batch_size):
                    clips = location_clips[b]
                    if clips.size(0) > 0:
                        clip_embeddings = self.model.audio_cnn(clips)
                        location_embedding = clip_embeddings.mean(dim=0)
                    else:
                        location_embedding = torch.zeros(self.model.embedding_dim, device=demographics.device)
                    batch_location_embeddings.append(location_embedding)
                
                batch_location_tensor = torch.stack(batch_location_embeddings)
                location_embeddings.append(batch_location_tensor)
            
            demographic_embeddings = self.model.demographic_mlp(demographics)
            all_embeddings = location_embeddings + [demographic_embeddings]
            combined_embeddings = torch.stack(all_embeddings, dim=1)
            
            attended_embeddings, attn_weights = self.model.attention(combined_embeddings)
            attention_weights_storage.extend(attn_weights.detach().cpu().numpy())
            
            flattened = attended_embeddings.view(batch_size, -1)
            output = self.model.classifier(flattened)
            return output
        
        self.model.forward = forward_with_attention
        
        # Process samples
        location_names = ['AV', 'PV', 'TV', 'MV', 'Demographics']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        with torch.no_grad():
            for idx, sample_idx in enumerate(sample_indices[:6]):
                if idx >= len(axes):
                    break
                
                # Prepare single sample
                single_location_data = {}
                for loc in ['AV', 'PV', 'TV', 'MV']:
                    single_location_data[loc] = [self.cached_samples[loc][sample_idx].unsqueeze(0)]
                
                single_demographics = self.cached_demographics[sample_idx:sample_idx+1]
                
                # Forward pass
                _ = self.model(single_location_data, single_demographics)
                
                # Get attention weights
                attn = attention_weights_storage[idx]
                if len(attn.shape) == 3:
                    attn = attn.mean(axis=0)[0]
                elif len(attn.shape) == 2:
                    attn = attn[0]
                
                # Plot
                sns.heatmap(attn.reshape(1, -1), 
                           annot=True, fmt='.3f', cmap='viridis',
                           xticklabels=location_names, yticklabels=['Attention'],
                           ax=axes[idx], cbar_kws={'shrink': 0.8})
                
                true_label = "Normal" if self.cached_labels[sample_idx] == 0 else "Abnormal"
                pred_label = "Normal" if self.cached_predictions[sample_idx] == 0 else "Abnormal"
                
                axes[idx].set_title(f'Sample {sample_idx}\nTrue: {true_label}, Pred: {pred_label}')
        
        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Attention Weights Visualization', fontsize=16, y=1.02)
        plt.show()
        
        # Restore original forward
        self.model.forward = original_forward
        
        return attention_weights_storage
    
    def visualize_cnn_features(self, sample_idx=0, location='AV', layer_name='cnn_conv3'):
        """Visualize CNN feature maps for a specific sample and location"""
        if self.cached_samples is None:
            print("No cached samples! Call cache_samples() first.")
            return
        
        self.register_hooks()
        self.model.eval()
        
        # Get sample
        clips = self.cached_samples[location][sample_idx]
        if clips.size(0) == 0:
            print(f"No clips available for location {location}")
            return
        
        # Forward pass through CNN only
        with torch.no_grad():
            _ = self.model.audio_cnn(clips)
        
        # Get activations
        if layer_name not in self.activations:
            print(f"Layer {layer_name} not found in activations")
            print(f"Available layers: {list(self.activations.keys())}")
            return
        
        features = self.activations[layer_name]
        
        # Plot feature maps
        if len(features.shape) == 4:  # Conv layer: [batch, channels, height, width]
            num_filters = min(16, features.shape[1])  # Show up to 16 filters
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            
            for i in range(num_filters):
                feature_map = features[0, i].cpu().numpy()  # First clip, i-th filter
                
                im = axes[i].imshow(feature_map, cmap='viridis', aspect='auto')
                axes[i].set_title(f'Filter {i}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], shrink=0.8)
            
            # Hide unused subplots
            for i in range(num_filters, len(axes)):
                axes[i].set_visible(False)
                
            plt.suptitle(f'CNN Features - {location} - {layer_name}', fontsize=16)
            plt.tight_layout()
            plt.show()
            
        elif len(features.shape) == 2:  # FC layer: [batch, features]
            feature_vector = features[0].cpu().numpy()  # First clip
            
            plt.figure(figsize=(12, 4))
            plt.plot(feature_vector)
            plt.title(f'Feature Vector - {location} - {layer_name}')
            plt.xlabel('Feature Index')
            plt.ylabel('Activation Value')
            plt.grid(True)
            plt.show()
        
        self.clear_hooks()
    
    def compare_features_by_class(self, layer_name='cnn_output', location='AV'):
        """Compare feature distributions between Normal and Abnormal cases"""
        if self.cached_samples is None:
            print("No cached samples! Call cache_samples() first.")
            return
        
        self.register_hooks()
        self.model.eval()
        
        normal_features = []
        abnormal_features = []
        
        with torch.no_grad():
            for i, label in enumerate(self.cached_labels):
                clips = self.cached_samples[location][i]
                if clips.size(0) > 0:
                    _ = self.model.audio_cnn(clips)
                    
                    if layer_name in self.activations:
                        features = self.activations[layer_name][0].cpu().numpy()  # First clip
                        
                        if label == 0:
                            normal_features.append(features)
                        else:
                            abnormal_features.append(features)
        
        if normal_features and abnormal_features:
            normal_features = np.array(normal_features)
            abnormal_features = np.array(abnormal_features)
            
            # Plot feature distributions
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Average features
            axes[0].plot(np.mean(normal_features, axis=0), label='Normal', alpha=0.7)
            axes[0].plot(np.mean(abnormal_features, axis=0), label='Abnormal', alpha=0.7)
            axes[0].set_title('Average Feature Values')
            axes[0].set_xlabel('Feature Index')
            axes[0].set_ylabel('Mean Activation')
            axes[0].legend()
            axes[0].grid(True)
            
            # Feature variance
            axes[1].plot(np.std(normal_features, axis=0), label='Normal', alpha=0.7)
            axes[1].plot(np.std(abnormal_features, axis=0), label='Abnormal', alpha=0.7)
            axes[1].set_title('Feature Variance')
            axes[1].set_xlabel('Feature Index')
            axes[1].set_ylabel('Std Activation')
            axes[1].legend()
            axes[1].grid(True)
            
            # Feature difference
            diff = np.mean(abnormal_features, axis=0) - np.mean(normal_features, axis=0)
            axes[2].plot(diff, color='red', alpha=0.7)
            axes[2].set_title('Abnormal - Normal Difference')
            axes[2].set_xlabel('Feature Index')
            axes[2].set_ylabel('Difference')
            axes[2].grid(True)
            axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            plt.suptitle(f'Feature Analysis - {location} - {layer_name}', fontsize=16)
            plt.tight_layout()
            plt.show()
        
        self.clear_hooks()
        return normal_features, abnormal_features
    
    def visualize_spectrograms(self, sample_indices=None, num_samples=4):
        """Visualize input spectrograms for different locations"""
        if self.cached_samples is None:
            print("No cached samples! Call cache_samples() first.")
            return
        
        if sample_indices is None:
            sample_indices = list(range(min(num_samples, len(self.cached_labels))))
        
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(16, 4*len(sample_indices)))
        if len(sample_indices) == 1:
            axes = axes.reshape(1, -1)
        
        locations = ['AV', 'PV', 'TV', 'MV']
        
        for row, sample_idx in enumerate(sample_indices):
            for col, location in enumerate(locations):
                clips = self.cached_samples[location][sample_idx]
                
                if clips.size(0) > 0:
                    # Take first clip and remove channel dimension
                    spectrogram = clips[0, 0].cpu().numpy()
                    
                    im = axes[row, col].imshow(spectrogram, cmap='viridis', aspect='auto', origin='lower')
                    axes[row, col].set_title(f'{location} - Sample {sample_idx}')
                    axes[row, col].set_xlabel('Time')
                    axes[row, col].set_ylabel('Mel Frequency')
                    plt.colorbar(im, ax=axes[row, col], shrink=0.8)
                else:
                    axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                      transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'{location} - Sample {sample_idx}')
                
                true_label = "Normal" if self.cached_labels[sample_idx] == 0 else "Abnormal"
                if col == 0:  # Add label only to first column
                    axes[row, col].set_ylabel(f'{true_label}\nMel Frequency')
        
        plt.tight_layout()
        plt.suptitle('Input Spectrograms by Location', fontsize=16, y=1.02)
        plt.show()

# Usage Examples and Quick Commands
def quick_visualization_setup(model, device, val_loader=None, test_loader=None):
    """Quick setup for visualization"""
    viz = ModelVisualizer(model, device, val_loader, test_loader)
    viz.cache_samples(num_samples=50)
    
    print("\n" + "="*60)
    print("QUICK VISUALIZATION COMMANDS:")
    print("="*60)
    print("# Attention weights:")
    print("viz.visualize_attention_weights([0,1,2,3,4,5])")
    print()
    print("# CNN features for specific sample/location:")
    print("viz.visualize_cnn_features(sample_idx=0, location='AV', layer_name='cnn_conv3')")
    print()
    print("# Compare features between classes:")
    print("normal_feat, abnormal_feat = viz.compare_features_by_class('cnn_output', 'AV')")
    print()
    print("# View input spectrograms:")
    print("viz.visualize_spectrograms([0,1,2,3])")
    print()
    print("# Available CNN layers: 'cnn_conv1', 'cnn_conv2', 'cnn_conv3', 'cnn_conv4', 'cnn_fc1', 'cnn_output'")
    print("="*60)
    
    return viz

print("Model Visualization Toolkit Ready!")
print("Usage: viz = quick_visualization_setup(model, device, val_loader)")