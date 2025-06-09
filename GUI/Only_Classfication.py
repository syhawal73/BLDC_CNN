# Import necessary libraries
import streamlit as st  # For creating the web app interface
import torch  # For deep learning model operations
import cv2  # For image processing tasks
import numpy as np  # For numerical operations on arrays
from torchvision import models, transforms  # For pre-trained models and image transformations
from PIL import Image  # For image handling
import torch.nn as nn  # For building and modifying neural network layers
import os  # For interacting with the operating system

# Disable Streamlit’s file watcher to prevent unnecessary inspections of PyTorch internal paths
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# Monkey-patch torch.classes.__path__ to avoid Streamlit inspection errors
torch.classes.__path__ = []

# Set the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define class names for classification
class_names = [
    "Banana Healthy Leaf",
    "Banana Insect Pest Disease",
    "Banana Moko Disease",
    "Banana Panama Disease",
    "Banana Sigatoka Disease",
]

# Dictionary containing causes and preventions for each disease
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

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance image contrast
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)  # Convert image to LAB color space
    l, a, b = cv2.split(lab)  # Split LAB channels
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    cl = clahe.apply(l)  # Apply CLAHE to L-channel
    limg = cv2.merge((cl, a, b))  # Merge channels back
    return cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)  # Convert back to RGB color space

# Define validation transformations for the input image
val_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert NumPy array to PIL Image
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize tensor with ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize tensor with ImageNet std
])

# Load the pre-trained model with specific layer freezing
@st.cache_resource  # Cache the model to avoid reloading on every run
def load_model():
    model_path = r"G:\Research - Banana Leaf Disease Classifcation\Kaggle\ResNet50_weights_only.pth"  # Path to the trained model

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet50 model

    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers initially

    for param in model.layer3.parameters():
        param.requires_grad = True  # Unfreeze layer3

    for param in model.layer4.parameters():
        param.requires_grad = True  # Unfreeze layer4

    in_feats = model.fc.in_features  # Get input features of the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Add dropout for regularization
        nn.Linear(in_feats, len(class_names))  # Final layer with output size equal to number of classes
    )

    for param in model.fc.parameters():
        param.requires_grad = True  # Unfreeze final fully connected layer

    model.load_state_dict(torch.load(model_path, map_location=device))  # Load trained weights
    model = model.to(device)  # Move model to the appropriate device
    model.eval()  # Set model to evaluation mode
    return model  # Return the loaded model

# Preprocess the input image before feeding it to the model
def preprocess_image(image):
    image_np = np.array(image)  # Convert PIL Image to NumPy array
    clahe_image = apply_clahe(image_np)  # Apply CLAHE to enhance contrast
    input_tensor = val_transform(clahe_image)  # Apply validation transformations
    input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    return input_batch  # Return the preprocessed image tensor

# Main function to run the Streamlit app
def main():
    st.title("Banana Leaf Disease Classifier")  # Set the title of the app
    st.write("Upload a clear image of a banana leaf for diagnosis")  # Instruction for the user

    model = load_model()  # Load the trained model

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])  # File uploader widget

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")  # Open and convert the uploaded image to RGB
        resized_image = image.resize((300, 300))  # Resize image for display

        col_center = st.columns([1, 2, 1])  # Create centered columns
        with col_center[1]:  # Use the middle column
            st.image(resized_image, caption="Uploaded Image")  # Display the uploaded image

        with st.spinner('Analyzing...'):  # Show a spinner while processing
            inputs = preprocess_image(image)  # Preprocess the image

            with torch.no_grad():
                outputs = model(inputs)  # Get model predictions

            probs = torch.nn.functional.softmax(outputs[0], dim=0)  # Apply softmax to get probabilities
            conf, pred_idx = torch.max(probs, 0)  # Get the highest probability and corresponding index
            predicted = class_names[pred_idx.item()]  # Get the predicted class name

        if predicted == "Banana Healthy Leaf":
            st.success("✅ Healthy Leaf Detected")  # Display success message
            st.write("No signs of disease found")  # Inform the user
        else:
            st.error(f"⚠️ Detected: {predicted}")  # Display error message with predicted disease
            st.write(f"Confidence: {conf.item()*100:.2f}%")  # Show confidence percentage

            st.subheader("Causes")  # Subheader for causes
            for cause in disease_info[predicted]["Causes"]:
                st.markdown(cause)  # Display each cause

            st.subheader("Preventions")  # Subheader for preventions
            for prevention in disease_info[predicted]["Preventions"]:
                st.markdown(prevention)  # Display each prevention

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
