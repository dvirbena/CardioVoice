# Cell 1: Data Loading and Preprocessing Functions


# This notebook implements a comprehensive ML model for classifying normal vs abnormal heart sounds
# using audio recordings from 4 auscultation locations and patient demographics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pandas as pd
import numpy as np
import librosa
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

CLIP_LENGTH = 3  # seconds

class AudioPreprocessor:
    """Handles audio preprocessing including bandpass filtering and mel-spectrogram generation"""
    
    def __init__(self, sample_rate=4000, n_mels=128, n_fft=1024, hop_length=512, 
                 normalization_method='minmax'): #zscore #minmax #robust #global_stats
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalization_method = normalization_method

    def bandpass_filter(self, audio, lowcut=25, highcut=800):
        """Apply bandpass filter to remove noise"""
        from scipy import signal
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def split_into_5s_clips(self, audio, sample_rate):
        """Split audio into 5-second clips, discarding remainder"""
        clip_length = CLIP_LENGTH * sample_rate
        n_clips = len(audio) // clip_length
        clips = []
        for i in range(n_clips):
            start = i * clip_length
            end = start + clip_length
            clips.append(audio[start:end])
        return clips
    
    def audio_to_melspec(self, audio):
        """Convert audio to mel-log spectrogram"""
        # Apply bandpass filter
        filtered_audio = self.bandpass_filter(audio)
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=filtered_audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

                # Apply normalization based on chosen method
        if self.normalization_method == 'zscore':
            # Z-score normalization (zero mean, unit variance) - Most common for CNNs
            log_mel_spec = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + 1e-8)
            
        elif self.normalization_method == 'minmax':
            # Min-Max normalization to [0, 1] range
            min_val = np.min(log_mel_spec)
            max_val = np.max(log_mel_spec)
            log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val + 1e-8)
            
        elif self.normalization_method == 'robust':
            # Robust normalization using percentiles (less sensitive to outliers)
            p5, p95 = np.percentile(log_mel_spec, [5, 95])
            log_mel_spec = np.clip((log_mel_spec - p5) / (p95 - p5 + 1e-8), 0, 1)
            
        elif self.normalization_method == 'global_stats':
            # Global normalization using dataset statistics (would need to be computed first)
            # This is a placeholder - you'd need to compute these stats from your training data
            # global_mean, global_std = compute_dataset_stats()  # Implement this function
            # log_mel_spec = (log_mel_spec - global_mean) / (global_std + 1e-8)
            pass
        
        # If normalization_method == 'none', no normalization is applied
        
        return log_mel_spec

    def compute_dataset_statistics(self, patients_data, num_samples=1000):
        """Compute global mean and std from a subset of the dataset for global normalization"""
        print("Computing dataset statistics for normalization...")
        
        all_spectrograms = []
        sample_count = 0
        
        for patient in tqdm(patients_data[:num_samples], desc="Computing stats"):
            if sample_count >= num_samples:
                break
                
            recordings = patient.get('recordings', {})
            for location in ['AV', 'PV', 'TV', 'MV']:
                if location in recordings:
                    for audio_file in recordings[location]:
                        try:
                            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                            clips = self.split_into_5s_clips(audio, sr)
                            
                            for clip in clips[:2]:  # Only use first 2 clips per file to save memory
                                # Apply preprocessing without normalization
                                temp_method = self.normalization_method
                                self.normalization_method = 'none'
                                mel_spec = self.audio_to_melspec(clip)
                                self.normalization_method = temp_method
                                
                                all_spectrograms.append(mel_spec)
                                sample_count += 1
                                
                                if sample_count >= num_samples:
                                    break
                            
                            if sample_count >= num_samples:
                                break
                        except Exception as e:
                            continue
                    
                    if sample_count >= num_samples:
                        break
                
                if sample_count >= num_samples:
                    break
        
        if all_spectrograms:
            # Concatenate all spectrograms and compute global statistics
            all_data = np.concatenate([spec.flatten() for spec in all_spectrograms])
            global_mean = np.mean(all_data)
            global_std = np.std(all_data)
            
            print(f"Dataset statistics computed from {len(all_spectrograms)} spectrograms:")
            print(f"Global mean: {global_mean:.4f}")
            print(f"Global std: {global_std:.4f}")
            
            return global_mean, global_std
        else:
            print("Warning: No spectrograms found for statistics computation")
            return 0.0, 1.0

def parse_patient_file(file_path):
    """Parse patient .txt file to extract demographics and labels"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    patient_info = {}
    # Parse first line: subject_id, num_recordings, sample_rate
    first_line = lines[0].strip().split()
    patient_info['subject_id'] = first_line[0]
    patient_info['num_recordings'] = int(first_line[1])
    patient_info['sample_rate'] = int(first_line[2])
    
    # Parse recording files info
    recordings = []
    for i in range(1, patient_info['num_recordings'] + 1):
        parts = lines[i].strip().split()
        recordings.append({
            'location': parts[0],
            'hea_file': parts[1],
            'wav_file': parts[2],
            'tsv_file': parts[3]
        })
    patient_info['recordings'] = recordings
    
    # Parse metadata
    for line in lines[patient_info['num_recordings'] + 1:]:
        if line.startswith('#'):
            key_value = line[1:].strip().split(': ', 1)
            if len(key_value) == 2:
                key, value = key_value
                if value != 'nan':
                    patient_info[key.lower().replace(' ', '_')] = value
                else:
                    patient_info[key.lower().replace(' ', '_')] = None
    
    return patient_info

def load_dataset(data_dir):
    """Load entire dataset and create structured dataframe"""
    data_dir = Path(data_dir)
    patients_data = []
    
    # Get all patient .txt files
    patient_files = list(data_dir.glob("*.txt"))
    
    for patient_file in tqdm(patient_files, desc="Loading patient data"):
        try:
            patient_info = parse_patient_file(patient_file)
            
            # Extract key demographic features
            demographics = {
                'subject_id': patient_info['subject_id'],
                'age': patient_info.get('age'),
                'sex': patient_info.get('sex'),
                'height': patient_info.get('height'),
                'weight': patient_info.get('weight'),
                'pregnancy_status': patient_info.get('pregnancy_status'),
                'outcome': patient_info.get('outcome'),  # Normal/Abnormal
                'murmur': patient_info.get('murmur')     # Present/Absent/Unknown
            }
            
            # Add recording file paths
            recordings_by_location = {}
            for rec in patient_info['recordings']:
                location = rec['location']
                if location in ['AV', 'PV', 'TV', 'MV']:  # Only use main 4 locations
                    if location not in recordings_by_location:
                        recordings_by_location[location] = []
                    recordings_by_location[location].append(data_dir / rec['wav_file'])
            
            demographics['recordings'] = recordings_by_location
            patients_data.append(demographics)
            
        except Exception as e:
            print(f"Error processing {patient_file}: {e}")
            continue
    
    return patients_data


class HeartSoundDataset(Dataset):
    """Dataset class for heart sound classification"""
    
    def __init__(self, patients_data, preprocessor=None, demographic_scaler=None):
        self.patients_data = patients_data
        self.preprocessor = preprocessor if preprocessor else AudioPreprocessor()
        self.demographic_scaler = demographic_scaler
        self.locations = ['AV', 'PV', 'TV', 'MV']
        
        # Filter out patients without outcome labels
        self.valid_patients = [p for p in patients_data if p.get('outcome') in ['Normal', 'Abnormal']]
        
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        patient = self.valid_patients[idx]

        # Process audio for each location
        location_embeddings = {}
        
        for location in self.locations:
            if location in patient.get('recordings', {}):
                # Get all recordings for this location
                recording_files = patient['recordings'][location]
                
                all_clips_embeddings = []
                
                for audio_file in recording_files:
                    try:
                        # Load audio
                        audio, sr = librosa.load(audio_file, sr=self.preprocessor.sample_rate)
                        
                        # Split into 5-second clips
                        clips = self.preprocessor.split_into_5s_clips(audio, sr)
                        
                        # Process each clip
                        for clip in clips:
                            mel_spec = self.preprocessor.audio_to_melspec(clip)
                            # Convert to tensor and add channel dimension
                            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
                            all_clips_embeddings.append(mel_tensor)
                            
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
                
                if all_clips_embeddings:
                    # Stack all clips for this location
                    location_embeddings[location] = torch.stack(all_clips_embeddings)
                else:
                    # Create dummy tensor if no valid clips
                    location_embeddings[location] = torch.zeros(1, 1, self.preprocessor.n_mels, 
                                                              self.preprocessor.n_mels)
            else:
                # Create dummy tensor for missing location
                location_embeddings[location] = torch.zeros(1, 1, self.preprocessor.n_mels, 
                                                          self.preprocessor.n_mels)
        
        # Process demographics
        demographics = self.process_demographics(patient)
        
        # Create label (1 for Abnormal, 0 for Normal)
        label = 1 if patient['outcome'] == 'Abnormal' else 0
        
        return location_embeddings, demographics, label
    
    def process_demographics(self, patient):
        """Process demographic information into numerical features"""
        # Extract and encode demographic features
        features = []
        
        # Height (convert to float or use median if missing)
        try:
            height = float(patient.get('height', 120.0))  # median height in dataset
        except (ValueError, TypeError):
            height = 120.0
        features.append(height)
        
        # Weight (convert to float or use median if missing)
        try:
            weight = float(patient.get('weight', 25.0))  # median weight in dataset
        except (ValueError, TypeError):
            weight = 25.0
        features.append(weight)
        
        # Sex (0 for Female, 1 for Male)
        sex = 1 if patient.get('sex') == 'Male' else 0
        features.append(sex)
        
        # Age (encode age categories as numerical)
        age_mapping = {
            'Neonate': 0, 'Infant': 1, 'Child': 2, 
            'Adolescent': 3, 'Young Adult': 4
        }
        age = age_mapping.get(patient.get('age'), 2)  # default to Child
        features.append(age)
        
        # Pregnancy status (0 for False/None, 1 for True)
        pregnancy = 1 if patient.get('pregnancy_status') == 'True' else 0
        features.append(pregnancy)
        
        demographics_tensor = torch.FloatTensor(features)
        
        # Apply scaling if scaler is provided
        # if self.demographic_scaler:
        #     demographics_tensor = torch.FloatTensor(
        #         self.demographic_scaler.transform(features.reshape(1, -1)).flatten()
        #     )

        if self.demographic_scaler:
            features_np = np.array(features).reshape(1, -1)
            demographics_tensor = torch.FloatTensor(
                self.demographic_scaler.transform(features_np).flatten()
            )
        
        return demographics_tensor