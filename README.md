# 📡 AMC-MLOps — Automatic Modulation Classification

[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/phishinchips67-lang/amc-signal-classifier)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub-orange)](https://dagshub.com/phishinchips67-lang/amc-mlops/experiments)
[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Automatic Modulation Classification (AMC) using deep learning on the RadioML 2016.10a dataset.
A 1D Convolutional Neural Network classifies 11 modulation types purely from raw I/Q samples,
tracked across 20 SNR levels using MLflow on DagsHub with dataset versioning via DVC.

---

## 🚀 Live Demo

Try the classifier directly in your browser — no installation, no coding needed.

👉 **[Launch Interactive Demo](https://huggingface.co/spaces/phishinchips67-lang/amc-signal-classifier)**

Generate a synthetic signal at any SNR level, or upload your own I/Q file and get an
instant classification with confidence scores.

---

## 📖 What is Automatic Modulation Classification?

In any wireless communication system, the transmitter encodes information by modifying
a carrier wave, a process called **modulation**. The receiver normally knows in advance
which modulation scheme is being used. But in many real-world scenarios, the receiver
must figure it out automatically:

- **Software Defined Radios (SDR)** that decode unknown signals
- **Cognitive radio** systems that adapt to the spectrum dynamically
- **Electronic warfare** and signals intelligence
- **Spectrum monitoring** by regulatory authorities

AMC is the task of identifying the modulation scheme purely from the received signal,
without any prior agreement between transmitter and receiver.

---

## 📡 The I/Q Signal Representation

Any bandpass wireless signal can be completely described by two components:

```
s(t) = I(t)·cos(2πft) − Q(t)·sin(2πft)
```

Where:
- **I(t)** — In-phase component (real axis)
- **Q(t)** — Quadrature component (imaginary axis)
- **f** — carrier frequency

Together, I and Q form a complex baseband representation of the signal. Every sample
in this project is a **(2, 128)** array of 2 rows (I and Q channels), each with 128
time samples. This is the raw input to the neural network, with no feature engineering.

---

## 📊 Dataset — RadioML 2016.10a

The **RadioML 2016.10a** dataset was created by DeepSig and is the standard benchmark
for deep learning based AMC research. It was generated using GNU Radio with realistic
channel impairments including:

- Multipath fading
- Local oscillator frequency offset and drift
- Additive White Gaussian Noise (AWGN)
- Sample rate offset

### Dataset Structure

```
{(modulation, SNR): array of shape (1000, 2, 128)}
```

| Property | Value |
|----------|-------|
| Modulation classes | 11 |
| SNR range | −20 dB to +18 dB (step 2 dB) |
| SNR levels | 20 |
| Samples per class per SNR | 1,000 |
| Total samples | 220,000 |
| Sample shape | (2, 128) — I/Q × time |
| Dataset size | ~55 MB |

### SNR Interpretation

| SNR | Signal Quality | Expected Model Accuracy |
|-----|---------------|------------------------|
| +18 dB | Very clean | ~90–95% |
| +10 dB | Good | ~80–85% |
| 0 dB | Moderate | ~60–70% |
| −10 dB | Noisy | ~30–40% |
| −20 dB | Buried in noise | ~15–20% (near random) |

---

## 📻 Modulation Types Explained

### Digital Modulations

**BPSK — Binary Phase Shift Keying**
Encodes 1 bit per symbol by shifting the carrier phase between 0° and 180°.
The signal lives entirely on the I axis. Used in GPS, deep space communications,
and Wi-Fi (802.11b). Most robust modulation, works at very low SNR.

**QPSK — Quadrature Phase Shift Keying**
Encodes 2 bits per symbol using four phase states: 45°, 135°, 225°, 315°.
Uses both I and Q channels equally. Used in satellite communications,
cable modems, and 3G/4G cellular networks.

**8PSK — 8-Phase Shift Keying**
Encodes 3 bits per symbol using eight equally spaced phase states.
More spectrally efficient than QPSK but requires higher SNR.
Used in satellite communications (DVB-S2).

**QAM16 — 16 Quadrature Amplitude Modulation**
Encodes 4 bits per symbol using a 4×4 grid of constellation points.
Varies both amplitude and phase. Used in 4G LTE downlink and Wi-Fi (802.11n).

**QAM64 — 64 Quadrature Amplitude Modulation**
Encodes 6 bits per symbol using an 8×8 grid of constellation points.
High spectral efficiency but requires clean channel (high SNR).
Used in 5G NR, cable TV (DOCSIS), and Wi-Fi (802.11ac).

**PAM4 — Pulse Amplitude Modulation 4**
Encodes 2 bits per symbol using four amplitude levels on the I axis only.
Used in high-speed optical and ethernet communications (400G ethernet).

**CPFSK — Continuous Phase Frequency Shift Keying**
Frequency shift keying with phase continuity between symbols, reducing
spectral splatter. Used in professional radio and telemetry systems.

**GFSK — Gaussian Frequency Shift Keying**
CPFSK with a Gaussian filter applied to smooth frequency transitions.
Used in **Bluetooth** Classic and Zigbee. Very power efficient.

### Analog Modulations

**AM-DSB — Amplitude Modulation Double Sideband**
The classic AM modulation. Information encoded in the amplitude envelope.
Both sidebands (upper and lower) are transmitted. Used in AM broadcast radio.

```
s(t) = [1 + m·x(t)] · cos(2πft)
```

Where x(t) is the message signal and m is the modulation index.

**AM-SSB — Amplitude Modulation Single Sideband**
Like AM-DSB but with one sideband suppressed, halving the bandwidth.
More efficient but harder to demodulate. Used in aviation and amateur radio.

**WBFM — Wideband Frequency Modulation**
Information encoded in instantaneous frequency deviation of the carrier.
```
s(t) = cos(2πft + 2π·kf·∫x(τ)dτ)
```
Where kf is the frequency sensitivity. Used in **commercial FM broadcast radio**
(88–108 MHz). High audio quality, immune to amplitude noise.

---

## 🧠 Model — AMCNet

A 1D Convolutional Neural Network that treats I and Q as two input channels
and slides learned filters across 128 time samples — conceptually similar to
learned matched filtering.

```
Input: (batch, 2, 128)
       ↓
Conv1d(2→64)   + ReLU     ← learns low-level transitions
Conv1d(64→64)  + ReLU
MaxPool1d(2)              → (batch, 64, 64)
       ↓
Conv1d(64→128)  + ReLU    ← learns symbol-level patterns
Conv1d(128→128) + ReLU
MaxPool1d(2)              → (batch, 128, 32)
       ↓
Flatten → Linear(4096→256) + ReLU + Dropout(0.5)
       ↓
Linear(256→11)            ← one logit per modulation class
       ↓
Output: (batch, 11)
```

| Property | Value |
|----------|-------|
| Parameters | ~1.1M |
| Input shape | (2, 128) |
| Output classes | 11 |
| Optimizer | Adam (lr=1e-3) |
| Loss function | CrossEntropyLoss |
| Regularization | Dropout 0.5 |

---

## 📈 Results

### Accuracy vs SNR

![Accuracy vs SNR](https://raw.githubusercontent.com/phishinchips67-lang/amc-mlops/master/models/accuracy_vs_snr.png)

The model learns a clear S-curve relationship with SNR — high accuracy on clean
signals degrading gracefully as noise increases, consistent with information
theoretic limits of modulation distinguishability.

### Confusion Matrix at +18 dB SNR

![Confusion Matrix](https://raw.githubusercontent.com/phishinchips67-lang/amc-mlops/master/models/confusion_matrix_high_snr.png)

At high SNR the model performs excellently on most classes. Notable confusions:
- **QAM16 vs QAM64** — similar constellation structure, differ only in density
- **CPFSK vs GFSK** — both are continuous phase FSK variants
- **AM-DSB vs WBFM** — both show broad spectral spread at baseband

These confusions reflect genuine signal similarity, not model failure.

---

## 🔬 MLflow Experiments

Three experiments tracked across different SNR conditions:

| Experiment | SNR Range | Epochs | Val Accuracy |
|------------|-----------|--------|-------------|
| High SNR | 0 to +18 dB | 20 | ~92% |
| Low SNR | −20 to −2 dB | 20 | ~28% |
| All SNR | −20 to +18 dB | 25 | ~60% |

[📊 View all runs on DagsHub →](https://dagshub.com/phishinchips67-lang/amc-mlops/experiments)

---

## 🛠️ MLOps Stack

| Tool | Purpose |
|------|---------|
| **PyTorch** | Model definition and training |
| **MLflow** | Experiment tracking — logs params, metrics, artifacts per run |
| **DagsHub** | Hosted MLflow UI — compare runs across SNR experiments |
| **DVC** | Dataset versioning — 55MB .pkl tracked via pointer file |
| **Google Drive** | DVC remote storage |
| **Gradio** | Interactive demo interface |
| **Hugging Face Spaces** | Free demo hosting |

---

## 🔁 Reproduce

```bash
# Clone repo
git clone https://github.com/phishinchips67-lang/amc-mlops
cd amc-mlops

# Install dependencies
pip install -r requirements.txt

# Pull dataset via DVC
dvc pull

# Train all 3 experiments
python src/train.py

# Evaluate and generate plots
python src/evaluate.py
```

---

## 📁 Project Structure

```
amc-mlops/
├── src/
│   ├── dataset.py          # RadioML data loading, filtering by SNR
│   ├── model.py            # AMCNet 1D CNN architecture (with BatchNorm)
│   ├── train.py            # Training loop + MLflow logging
│   └── evaluate.py         # Per-SNR evaluation + plot generation
├── models/
│   ├── amc_all.pt          # Trained model weights (all SNR)
│   ├── amc_0to18.pt        # High SNR model
│   ├── amc_-20to-2.pt      # Low SNR model
│   ├── accuracy_vs_snr.png
│   └── confusion_matrix_high_snr.png
├── data/
│   └── RML2016.10a_dict.pkl.dvc   # DVC pointer (not the actual data)
├── .dvc/config                     # DVC remote config
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📚 References

-T.J. O'Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," IEEE Trans. Cognitive Commun. Netw., 2017

-RadioML Dataset — DeepSig Inc. (opendata.deepsig.io)

-PyTorch Documentation — pytorch.org

---

## 👤 Author

**phishinchips67-lang**
ECE Graduate | Industrial Automation Engineer transitioning to Embedded + ML Systems

[![GitHub](https://img.shields.io/badge/GitHub-phishinchips67--lang-black)](https://github.com/phishinchips67-lang)
