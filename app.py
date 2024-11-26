import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model_arch import AlzheimerDetector

# Load your model
@st.cache_resource
def load_model():
    model= AlzheimerDetector(input_shape=3,hidden_units=10,output_shape=4,image_dimension=128).to("cpu")
    model.load_state_dict(torch.load(f"models/alz_CNN.pt", weights_only=True,map_location=torch.device('cpu') ))
    model.eval()
    return model

model = load_model()

# Define the class names
class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to match the input size of your model
    transforms.Lambda(lambda x: x.convert("RGB")),  # Convert image to RGB
    transforms.ToTensor(),          # Convert image to tensor
])

# Streamlit App
st.title("Alzheimer Detector using Brain MRI Images")
st.write("Upload a brain MRI image to detect Alzheimer's stage.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)  # Convert to grayscale
    st.image(image, caption="Uploaded Image",  use_container_width=True)

    # Preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)

    # Predict the class
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)  # Get the index of the highest score
        predicted_class = class_names[predicted.item()]

    # Display the result
    st.write(f"### Prediction: {predicted_class}")
