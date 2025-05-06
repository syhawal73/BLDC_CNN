import os
# Disable Streamlit’s auto file‑watcher to avoid inspecting PyTorch internal paths
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
# Monkey‑patch out the bogus torch.classes.__path__
torch.classes.__path__ = []

import streamlit as st
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np


# ----------------- Configuration & Data -----------------

# Define class names for your prediction
class_names = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Sigatoka Disease",
]

# Define disease info (causes and preventions)
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
            "\ni] Contaminated soil or water.\n",
            "\nii] Movement of infected plant material.\n",
            "\n5] Symptoms include yellowing of leaves, wilting, and black streaks inside the stem.\n"
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

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----------------- Model Definition & Loading -----------------

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze most layers and fine-tune only layer3 & layer4
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Update the model path to point to your model file
model_path = r"G:\Mega_Project\Models\ResNet50_HKC_5_Classes\ResNet50_HKC_5_L34.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ----------------- Helper Functions -----------------

def predict_disease_with_model(image):
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, pred_idx = torch.max(probabilities, dim=1)
        return class_names[pred_idx.item()]

def load_reference_image():
    ref_path = r"G:\Mega_Project\Dataset\5_Classes_TVT\train\Banana Healthy Leaf\Image_4604.jpg"
    if os.path.exists(ref_path):
        return Image.open(ref_path).resize((250, 250))
    return Image.new("RGB", (250, 250), color="green")

def highlight_non_green(original, grayscale, alpha=0.5):
    """
    Blends red on non-green portions of the grayscale image.
    A pixel is 'green' if its G-channel ≥ R and ≥ B.
    """
    arr_original = np.array(original)
    arr_gray = np.array(grayscale).astype(np.float32)
    mask_green = (arr_original[:, :, 1] >= arr_original[:, :, 0]) & (arr_original[:, :, 1] >= arr_original[:, :, 2])
    red_overlay = np.array([255, 0, 0], dtype=np.float32)
    blend = np.where(
        np.expand_dims(~mask_green, axis=2),
        alpha * red_overlay + (1 - alpha) * arr_gray,
        arr_gray
    )
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    return Image.fromarray(blend)

# Preload reference images
ref_image = load_reference_image()
ref_gray = ref_image.convert("L").convert("RGB")

# ----------------- Streamlit Interface -----------------

st.title("Banana Leaf Disease Detector")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Center display
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.image(image.resize((250, 250)), caption="Uploaded Image")

    # Prediction
    prediction = predict_disease_with_model(image)
    st.success(f"Prediction: **{prediction}**")

    # Highlight if diseased
    user_gray = image.convert("L").convert("RGB").resize((250, 250))
    if prediction != "Banana Healthy Leaf":
        user_gray = highlight_non_green(image.resize((250, 250)), user_gray)

    # Show reference vs. user
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Healthy Reference")
        st.image(ref_image, caption="Reference (Color)")
        st.image(ref_gray, caption="Reference (Grayscale)")
    with col_right:
        st.subheader("Your Leaf")
        st.image(image.resize((250, 250)), caption="Your Image (Color)")
        st.image(user_gray, caption="Your Image (Gray/Highlight)")

    # Disease details
    if prediction != "Banana Healthy Leaf":
        info = disease_info.get(prediction, {})
        st.subheader("Disease Details")
        st.markdown("**Causes:**")
        st.markdown("".join(info.get("Causes", ["N/A"])))
        st.markdown("**Preventions:**")
        st.markdown("".join(info.get("Preventions", ["N/A"])))
    else:
        st.info("No disease detected. The leaf looks healthy!")
