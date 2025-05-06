import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

import os
# Disable Streamlit’s auto file‑watcher to avoid inspecting PyTorch internal paths
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
# Monkey‑patch out the bogus torch.classes.__path__
torch.classes.__path__ = []

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names and disease information
class_names = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease", 
    "Banana Panama Disease",
    "Banana Sigatoka Disease",
]

disease_info = {
    "Banana Insect Pest Disease": {
        "Causes": [
            "\nDamage caused by insects such as:\n",
            "\ni] Banana weevils (Cosmopolites sordidus): Adults bore into the plant's stem, weakening the plant and reducing yield.\n",
            "\nii] Nematodes: Microscopic worms that attack the roots, causing stunted growth and poor nutrient uptake.\n",
            "\niii] Aphids: These sap-sucking pests weaken plants and may also transmit viruses like Banana Bract Mosaic Virus.\n"
        ],
        "Preventions": [
            "\n1) Apply insecticides to control pests.\n",
            "\n2) Regularly inspect plants for early signs of infestation.\n",
        ],
    },
    "Banana Moko Disease": {
        "Causes": [
            "\n1] Moko disease is caused by the bacterium 'Ralstonia solanacearum.'\n",
            "\n2] It spreads through:\n",
            "\ni] Infected tools used for pruning or harvesting.\n",
            "\nii] Contaminated soil or water.\n",
            "\niii] Insects that come into contact with infected flowers.\n",
            "\n3] Symptoms include wilting of leaves, yellowing, and internal discoloration of the fruit and stem.\n"
        ],
        "Preventions": [
            "\n1) Sanitize tools before use.\n",
            "\n2) Avoid planting in areas with known infections.\n",
            "\n3) Remove and destroy infected plants immediately.\n"
        ],
    },
    "Banana Panama Disease": {
        "Causes": [
            "\n1] Caused by the soil-borne fungus 'Fusarium oxysporum cubense (Foc)'.\n",
            "\n2] This disease attacks the banana plant's roots and vascular system, cutting off water and nutrient flow.\n",
            "\n3] There are different strains of the fungus, with Tropical Race 4 (TR4) being the most devastating.\n",
            "\n4] It spreads through:\n",
            "\ni] Contaminated soil or water.",
            "\nii] Movement of infected plant material.",
            "\niii] Contaminated equipment.\n",
            "\n5] Symptoms include yellowing of leaves, wilting, and black streaks inside the stem."
        ],
        "Preventions": [
            "\n1) Use disease-resistant banana varieties.\n",
            "\n2) Practice crop rotation and maintain good field sanitation.\n",
        ],
    },
    "Banana Sigatoka Disease": {
        "Causes": [
            "\n1] Sigatoka is caused by fungal pathogens 'Mycosphaerella fijiensis' & 'Mycosphaerella musicola.'\n",
            "\n2] The fungi infect banana leaves, forming dark streaks or spots that merge and lead to leaf death.\n",
            "\n3] These fungi thrive in warm, humid environments and spread via windborne spores.\n"
        ],
        "Preventions": [
            "\n1) Remove and destroy infected leaves.\n",
            "\n2) Apply fungicides regularly as a preventive measure.\n",
            "\n3) Ensure proper plant spacing to allow air circulation and reduce humidity.\n"
        ],
    },
}

# CLAHE preprocessing function
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)

# Validation transforms
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Model loading with proper layer freezing
@st.cache_resource
def load_model():
    model_path = r"G:\Mega_Project\Updated_Models\BDD_CNN_ResNet50.pth"
    
    # Create model with original architecture
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze layers as in training
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer3, layer4, and fc
    for param in model.layer3.parameters():
        param.requires_grad = True
        
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Modify FC layer (fixed the missing parenthesis)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, len(class_names))
    )  # Added closing parenthesis here
    
    # Unfreeze FC parameters
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    image_np = np.array(image)
    clahe_image = apply_clahe(image_np)
    input_tensor = val_transform(clahe_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    return input_batch

# Streamlit UI
# ... (previous imports and code remain the same)

# Streamlit UI
def main():
    st.title("Banana Leaf Disease Classifier")
    st.write("Upload a clear image of a banana leaf for diagnosis")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Resize image 
        resized_image = image.resize((300, 300))
        
        # Create centered columns
        col_center = st.columns([1, 2, 1])
        with col_center[1]:  # Use the middle column
            st.image(resized_image, caption="Uploaded Image")
        
        with st.spinner('Analyzing...'):
            inputs = preprocess_image(image)  # Process original image for model
            
            with torch.no_grad():
                outputs = model(inputs)
            
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, pred_idx = torch.max(probs, 0)
            predicted = class_names[pred_idx.item()]
        
        if predicted == "Banana Healthy Leaf":
            st.success("✅ Healthy Leaf Detected")
            st.write("No signs of disease found")
        else:
            st.error(f"⚠️ Detected: {predicted}")
            st.write(f"Confidence: {conf.item()*100:.2f}%")
            
            # Display causes and preventions
            st.subheader("Causes")
            for cause in disease_info[predicted]["Causes"]:
                st.markdown(cause)
            
            st.subheader("Preventions")
            for prevention in disease_info[predicted]["Preventions"]:
                st.markdown(prevention)

if __name__ == "__main__":
    main()