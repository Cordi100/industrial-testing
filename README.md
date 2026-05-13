# Industrial Visual Inspection System (InspectAI) 🚀

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![AI-PM](https://img.shields.io/badge/Focus-AI--PM-orange.svg)

An end-to-end industrial surface defect detection system based on the **PatchCore** algorithm. Designed specifically for **AI Product Management (AI-PM)** portfolios to demonstrate technical depth and business impact in manufacturing scenarios (e.g., AOI/Quality Control).

## 🌟 Key Features
- **Unsupervised Anomaly Detection**: Requires only "Good" samples for training—ideal for industrial scenarios where defect samples are rare.
- **SOTA Performance**: Achieves 100% AUROC and **0.0% False Reject Rate (FRR)** on the MVTec AD bottle dataset.
- **Pixel-level Localization**: Generates heatmaps to accurately pinpoint defects (scratches, holes, contamination).
- **Premium Dashboard**: A dark-mode, high-contrast web interface for real-time inference and yield rate monitoring.

## 🛠 Tech Stack
- **AI Core**: Python, TensorFlow (ResNet-50 Feature Extraction), Numpy (Coreset Memory Bank), Scikit-Learn.
- **Backend**: FastAPI, Uvicorn (Asynchronous API).
- **Frontend**: HTML5, Vanilla CSS (Premium Dark Mode), JavaScript (ES6).
- **Dataset**: MVTec AD (Industrial Standard).

## 📊 Business Metrics (PM Focus)
| Metric | Result | Impact |
| :--- | :--- | :--- |
| **FAR (False Accept Rate)** | 10.0% | Overhead cost for secondary manual review. |
| **FRR (False Reject Rate)** | **0.0%** | Zero customer complaints; high product safety. |
| **AUROC** | 100% | High discriminatory power between good and bad items. |

## 🚀 Quick Start

### 1. Environment Setup
```bash
git clone https://github.com/your-username/industrial-ml.git
cd industrial-ml
pip install -r requirements.txt
```

### 2. Training
Place the MVTec bottle dataset in `data/mvtec_ad/bottle`.
```bash
python -m src.patchcore
```

### 3. Launch Web UI
```bash
python -m uvicorn web.app:app --reload
```
Visit `http://localhost:8000` to start inspecting!

## 📂 Project Structure
```text
industrial_ML/
├── data/           # (Excluded from Git) MVTec AD Dataset
├── models/         # (Excluded from Git) Saved Model Weights (.pkl)
├── src/            # AI Core Logic (PatchCore, Evaluation)
├── web/            # Dashboard (FastAPI, Static Assets)
├── README.md       # Project Documentation
└── requirements.txt
```
