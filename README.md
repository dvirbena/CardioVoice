# 00460217 - CardioVoice: Detecting Cardiovascular Risks from Heart Sounds  

This project aims to develop an automated, non-invasive heart murmur detection system using pediatric phonocardiogram (PCG) recordings and demographic data. The model combines CNN-based spectrogram analysis with multi-head attention and demographic features to predict clinical outcomes, even in noisy, real-world environments.

![CardioVoice Diagram](Figures/cardiovoice_diagram.png)

---

## Table of Contents  
1. [Overview](#1-overview)  
2. [How To Run](#2-how-to-run)  
3. [Datasets](#3-datasets)  
4. [Challenges and Solutions](#4-challenges-and-solutions)  
5. [Model Architecture](#5-model-architecture)  
   - [5.1 CNN for Spectrograms](#51-cnn-for-spectrograms)  
6. [Training Approach](#6-training-approach)  
7. [Better Approach](#7-better-approach)  
8. [Training Details](#8-training-details)  
9. [Preliminary Results](#9-preliminary-results)  
10. [Conclusions](#10-conclusions)  

---

## 1. Overview  
CardioVoice is a deep learning system for detecting heart murmurs and predicting clinical outcomes from pediatric PCG recordings. Recordings are taken from multiple auscultation locations (AV, PV, TV, MV) and combined with patient demographics.  

**Motivation:**  
- Congenital and acquired heart disease is a significant health issue.  
- Early detection improves outcomes and reduces costs.  
- Many regions lack cardiology expertise.  
- Automated analysis enables point-of-care diagnosis.  

---

## 2. How To Run  

### Files Explanation  
**Code folder:**  
- **CardioVoice.ipynb** – Complete notebook version of the project, recommended for Google Colab.  

**Python scripts:**  
We divided the coding cells into `.py` files if you prefer running the project this way. The order is as follows:  
1. `data_loading_and_preprocessing.py` – Data reading, filtering, segmentation, and feature extraction.  
2. `model.py` – Contains CNN, MLP, and Attention modules.  
3. `training_funcs.py` – Functions for loss calculation, evaluation, and metrics.  
4. `training_pipeline.py` – Full training and validation loop.  
5. `model_visualization_toolkit.py` – Plots for loss curves, accuracy, and confusion matrix.  
6. `usage.py` – Example of running inference on new data.  

**Additional Files:**  
- **Dataset.md** – Dataset download links and metadata.  

### Steps to Run  
1. Upload the dataset file into the notebook.  
2. (Optional) Mount Google Drive in Google Colab for persistent storage.
3. Set your data path correctly in usage.py
4. Run all notebook cells to preprocess, train, and evaluate the model.  

### Custom Dataset Generation  
You can modify the preprocessing scripts to handle new datasets by changing:  
- File paths in `data_loading_and_preprocessing.py`  
- Metadata mapping for demographics  

---

## 3. Datasets  
**Source:** [PhysioNet George B. Moody Challenge 2022](https://physionet.org/content/circor-heart-sound/1.0.3/)  
- **Size:** 5272 recordings, 1568 subjects, aged 0–21 years.  
- **Recording Locations:** AV, PV, TV, MV.  
- **Labels:** Murmur (Present/Absent/Unknown), Clinical Outcome (Normal/Abnormal).  

**Preprocessing:**  
- Bandpass filter: 25–800 Hz  
- Segmentation: 5-second clips (discard remainders)  
- Features: Log mel-spectrograms (`n_mels=128`, `n_fft=1024`, `hop_length=512`)  
- Demographics: Encoded and normalized (height, weight, sex, age, pregnancy status)  
- Missing locations → zero-padded tensors  

---

## 4. Challenges and Solutions  

| Challenge | Solution |
|-----------|----------|
| Noisy, multi-location PCG recordings | Bandpass filtering and multi-location fusion with attention |
| Missing auscultation locations | Zero-padding of missing site spectrograms |
| Imbalanced datasets | Stratified train/validation/test split by patient ID |
| Combining audio and tabular data | Parallel CNN (audio) + MLP (demographics) embeddings |

---

## 5. Model Architecture  

### Core Components  
- **AudioCNN:** 4 convolutional blocks → 128-D embedding.  
- **DemographicMLP:** 3 dense layers → 128-D embedding.  
- **Multi-head attention:** Aggregates location embeddings.  
- **Fusion & classification:** Concatenate embeddings → fully connected → output.  

---

### 5.1 CNN for Spectrograms  
- Conv2D → BatchNorm → ReLU → MaxPooling → Dropout  
- Repeated for 4 blocks to progressively reduce spatial dimensions and learn robust frequency-time patterns.  

---

## 6. Training Approach  
- Loss: Cross-Entropy  
- Optimizer: Adam (tuned learning rate)  
- Regularization: Dropout in CNN and MLP layers  
- Stratified splitting to avoid patient overlap between sets  

---

## 7. Better Approach  
**Audio + Demographics + Multi-head Attention:**  
- CNN extracts features per auscultation site.  
- Demographic MLP adds patient-level info.  
- Attention layer combines multi-location embeddings.  

---

## 8. Training Details  
- **Epochs:** Tuned with early stopping  
- **Batch size:** Configurable (default 32)  
- **Learning rate:** Optimized with scheduler  
- **Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix  

---

## 9. Preliminary Test Results  

| Model                   | AUC      | Accuracy | F1-score |
|-------------------------|----------|----------|----------|
| CardioVoice             | 0.6093   | 0.5608   | 0.6029   |

**Observations:**  
- Demographics boost performance.  
- Attention improves fusion of multiple locations.  
- Zero-padding for missing data is effective.  

**Results Visuals:**  
- ![Training and Validation Curves](Figures/output.png)  
- ![Confusion Matrix](Figures/output_confusion.png)  

---

## 10. Conclusions
In this work, we explored a novel fusion-based CNN + MLP + Attention architecture for pediatric heart sound analysis, integrating multi-location auscultation signals with demographic features. While the final accuracy did not surpass some simpler baselines, our approach demonstrates a new way of handling missing-location data and combining heterogeneous inputs in a unified end-to-end model.

The results suggest that this architecture is capable of learning meaningful patterns across both acoustic and demographic modalities, but also highlight the difficulty of the pediatric heart sound classification task—especially with limited, noisy, and imbalanced datasets. Importantly, the methodology developed here can serve as a foundation for future improvements, rather than an endpoint.

**Future Work:**
- CRNN or Transformer-based temporal modeling
- Self-supervised pretraining
- Lightweight deployment-ready models

