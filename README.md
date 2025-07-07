
# 🧠 Brain Tumor MRI Image Classification

This project builds an AI-based solution to classify brain MRI scans into four tumor categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**. It uses transfer learning with a fine-tuned **EfficientNetB0** model and provides an interactive **Streamlit** app for real-time predictions.

---

## 🚀 Project Highlights

- ✅ Fine-tuned **EfficientNetB0** with PyTorch
- 🎯 Test Accuracy: **96%**
- 🧠 Supports 4 tumor classes
- 🧪 Robust to class imbalance (via Focal Loss)
- 🔁 Enhanced generalization (RandAugment + Vertical Flip)
- 🌐 Real-time predictions using **Streamlit Web App**

---

## 📊 Model Performance

| Class       | Precision | Recall | F1-Score |
|-------------|-----------|--------|----------|
| Glioma      | 0.99      | 0.97   | 0.98     |
| Meningioma  | 0.92      | 0.97   | 0.95     |
| No Tumor    | 0.98      | 0.90   | 0.94     |
| Pituitary   | 0.96      | 1.00   | 0.98     |
| **Overall** | **0.96**  | **0.96** | **0.96** |

---

## 🧠 Dataset

- 4 categories: **glioma**, **meningioma**, **no_tumor**, **pituitary**
- Images are grayscale/RGB, resized to **224×224**
- Pre-split into `train/`, `valid/`, `test/` folders

> Dataset Source: Public brain MRI dataset (Kaggle / medical archive)

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — model training
- **Torchvision** — transforms & pretrained EfficientNet
- **Streamlit** — UI for prediction
- **Scikit-learn** — evaluation metrics

---

## 📁 Project Structure

```
brain-tumor-classifier/
├── brain_tumor_app.py           # Streamlit app
├── efficientnet_b0_final.pt     # Trained model file
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
└── data/                        # (Optional) Dataset structure
```

---

## 🔧 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Install Requirements
Manually:

```bash
pip install streamlit torch torchvision pillow scikit-learn matplotlib numpy
```

### 3. Launch the App

```bash
streamlit run brain_tumor_app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Deployment Options

- ✅ Run locally via `streamlit run`
- ☁️ [Streamlit Cloud](https://streamlit.io/cloud)
- ☁️ [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 📦 Model Details

- Model: **EfficientNetB0**
- Fine-tuned layers: `features.6` and `classifier`
- Loss: **Focal Loss** with meningioma weight boost
- Augmentation: **RandAugment + Vertical Flip**

---

## 🙌 Acknowledgements

- EfficientNetB0 from [torchvision.models](https://pytorch.org/vision/stable/models.html)
- Streamlit for rapid deployment
- Brain MRI dataset from public medical repositories

---

## Author
 - Yugeshwar V
