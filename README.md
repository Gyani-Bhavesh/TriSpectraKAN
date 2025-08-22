# TriSpectraKAN
Enhanced COPD detection via novel framework called KAN (Kolmogorov Arnold Network)


# TriSpectraKAN

**Enhanced COPD detection via a novel framework: the Kolmogorov–Arnold Network (KAN)**

---

##  Overview

TriSpectraKAN is a hybrid machine learning framework designed to diagnose Chronic Obstructive Pulmonary Disease (COPD) using lung sound analysis. It combines multiple audio feature extractors—Mel-frequency Cepstral Coefficients (MFCCs), Chromagram, and Mel Spectrograms—with a Kolmogorov–Arnold Network (KAN) fusion model for accurate, real-time detection.

**Key highlights**:
- Utilizes multi-modal audio features extracted from lung sound recordings.
- Achieves high performance: ~93% accuracy, ~97% precision, recall, and ~98% F1-score for COPD detection. :contentReference[oaicite:0]{index=0}
- Deployed successfully on Raspberry Pi, demonstrating real-time, low-resource feasibility. :contentReference[oaicite:1]{index=1}

---

##  Citation

If you use this code, please cite:



---

##  Paper Overview

Published on **21 February 2025**, this study presents TriSpectraKAN, leveraging three distinct audio features (MFCC, Chromagram, Mel Spectrogram) and fusing them via a hybrid KAN-based neural network to achieve robust COPD classification. :contentReference[oaicite:3]{index=3} The pipeline:
1. Audio preprocessing (resampling to 22.05 kHz, segmentation to 6 s clipped/padded units).
2. Feature extraction via CNN heads for each audio representation.
3. Fusion through a KAN-based architecture using splines as learnable activation functions. :contentReference[oaicite:4]{index=4}
4. Extensive evaluation across public datasets, performance metrics, real-world Raspberry Pi deployment, and validation on clinical recordings. :contentReference[oaicite:5]{index=5}

---

##  Repository Structure

The GitHub repo includes the following notebooks:

- `part-1-preprocessing.ipynb`  
  - Audio standardization: resampling, segmentation, and fixed-length clipping/padding to 6 s per sample.
- `part-2-handel-imbalance-creating-spectrogram.ipynb`  
  - Addresses class imbalance and extracts feature representations: MFCC, Chromagram, and Mel Spectrograms.
- `part-3-modelling-training.ipynb`  
  - Constructs the TriSpectraKAN model with KAN fusion architecture and CNN heads; includes model training using collected datasets.
- `copd-kan2 (1).ipynb`  
  - Likely an experimental or updated version of the KAN model—potentially refines architecture or hyperparameters.

---

##  Getting Started

### Prerequisites
- Python 3.8+
- Core libraries:
  - `numpy`, `librosa`, `torch` (PyTorch)
  - ML utilities: e.g., `scikit-learn`, `matplotlib`
  - Jupyter Notebook environment

### Setup & Usage
1. **Clone the repo**:  
   ```bash
   git clone https://github.com/Gyani-Bhavesh/TriSpectraKAN.git
   cd TriSpectraKAN
