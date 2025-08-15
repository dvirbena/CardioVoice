# 00460217 - CardioVoice: Detecting Cardiovascular Risks from Heart Sounds

![CardioVoice Diagram](Figures/cardiovoice_diagram.png)

## Introduction
CardioVoice is a deep learning system for detecting heart murmurs and predicting clinical outcomes from pediatric phonocardiogram (PCG) recordings. The recordings are collected from multiple auscultation locations and combined with demographic patient data.

## Project Goal
Develop an automated, non-invasive murmur detection model that can operate in noisy, real-world environments and support screening in areas with limited access to cardiology specialists.

## Motivation
- Congenital and acquired heart disease is a significant health issue.
- Early detection improves outcomes and reduces costs.
- Many regions lack cardiology expertise.
- Automated analysis enables point-of-care diagnosis.

## Related Work
- CNN-based spectrogram classification (PhysioNet 2016 Challenge).
- PhysioNet 2022 Challenge: multi-location PCG with demographics.
- This project combines spectrogram-based CNNs, demographic MLPs, and multi-head attention for multi-location signal fusion.

## Method

### Data Preprocessing
- **Bandpass filter:** 25–800 Hz
- **Segmentation:** 5-second clips (discard remainders)
- **Feature extraction:** Log mel-spectrograms (`n_mels=128`, `n_fft=1024`, `hop_length=512`)
- **Demographics:** Encode height, weight, sex, age category, pregnancy status; normalize values
- Missing auscultation locations filled with zero tensors

### Dataset Class Output
- Spectrogram tensor for each location (AV, PV, TV, MV)
- Demographic feature tensor
- Binary label (Normal = 0, Abnormal = 1)

### Model Architecture
- **AudioCNN:** 4 conv blocks → 128-D embedding
- **DemographicMLP:** 3 dense layers → 128-D embedding
- **Multi-head attention:** aggregates audio embeddings from four locations
- **Fusion & classification:** concatenated embedding → final output layer

### Training
- Loss: Cross-Entropy
- Optimizer: Adam (tuned LR)
- Regularization: Dropout in CNN and MLP layers

## Experiments

### Dataset
- **Source:** PhysioNet George B. Moody Challenge 2022  
  - [Dataset page](https://physionet.org/content/circor-heart-sound/1.0.3/)  
  - [Download link](https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/)  
- **Size:** 5272 recordings, 1568 subjects (ages 0–21)
- **Locations:** AV, PV, TV, MV
- **Labels:** Murmur (Present/Absent/Unknown), Clinical Outcome (Normal/Abnormal)

### Setup
- Stratified Train/Validation/Test split by patient ID
- Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Baselines: CNN (audio only), MLP (demographics only)

### Results

| Model                   | Accuracy | F1-score |
|-------------------------|----------|----------|
| CNN (audio only)        | 0.78     | 0.76     |
| MLP (demo only)         | 0.62     | 0.59     |
| CNN + MLP + Attention   | 0.84     | 0.83     |

Observations:
- Demographics improve classification performance.
- Multi-head attention helps when combining multiple locations.
- Zero-padding works for missing locations.

#### Training and Validation Curves
![Training and Validation Loss and Accuracy](Figures/output.png)

#### Confusion Matrix
![Confusion Matrix](Figures/output_confusion.png)

## Conclusion
The CNN+MLP+Attention architecture outperforms simpler baselines and works in noisy, multi-location PCG data.

## Future Work
- Temporal models (CRNN, Transformers) to capture sequential patterns
- Self-supervised pretraining on unlabeled PCG data
- Lightweight models for mobile/embedded deployment

## References
1. Liu, C. et al., PhysioNet/Computing in Cardiology Challenge 2016
2. Potes, C. et al., Ensemble methods for abnormal heart sound detection
3. Reyna, M.A. et al., PhysioNet Challenge 2022
4. Chollet, F., Xception: Deep Learning with Depthwise Separable Convolutions, CVPR 2017
