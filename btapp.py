# brain_tumor_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Define class names
class_names = ['Glioma', 'Meningioma', 'No_Tumor', 'Pituitary']

# Define image transforms (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load model
@st.cache_resource
def load_model():
    model = efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 4)
    )
    model.load_state_dict(torch.load("efficientnet_b0_final.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).squeeze()
        conf, pred = torch.max(probs, dim=0)
    return class_names[pred], conf.item(), probs.numpy()

# App UI
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Classifying..."):
            label, confidence, all_probs = predict(image)
            st.success(f"🎯 Predicted: **{label.upper()}** with {confidence * 100:.2f}% confidence")

            st.subheader("Confidence Scores:")
            for i, cls in enumerate(class_names):
                st.write(f"{cls.title()}: {all_probs[i]*100:.2f}%")
