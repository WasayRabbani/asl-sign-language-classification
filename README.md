# Real-Time American Sign Language (ASL) Word Classification

This project is a real-time computer vision system that recognizes and classifies **American Sign Language (ASL) words** using a webcam. It leverages **MediaPipe** for landmark extraction, **OpenCV** for video processing, and a **deep learning model (PyTorch)** for classification.

The system detects hand, pose, and facial landmarks in real time and predicts the performed ASL sign.

---

## 🚀 Features
- Real-time ASL word recognition using webcam
- MediaPipe Holistic landmark extraction
- Deep learning–based classification (PyTorch)
- Live prediction overlay on video stream
- Modular and easy-to-extend codebase

---

## 🛠️ Tech Stack
- **Python 3.10**
- **OpenCV**
- **MediaPipe**
- **PyTorch**
- **NumPy & Pandas**
- **scikit-learn**
- **Matplotlib / Seaborn**

---

---

## ⚙️ Installation

### 1️⃣ Create virtual environment
```bash
python -m venv venv

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

Install dependencies
pip install -r requirements.txt

Run the Application
python model_test.py

