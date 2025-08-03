import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import cv2

# Clear cache to ensure changes are reflected
st.cache_data.clear()
st.cache_resource.clear()

# Set page config
st.set_page_config(
    page_title="Gastric Cancer Classification",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Target ALL buttons including file upload */
    button {
        background-color: #0068c9 !important;
        color: white !important;
        border-color: #0068c9 !important;
    }
    
    button:hover {
        background-color: #0056b3 !important;
        color: white !important;
        border-color: #0056b3 !important;
    }
    
    /* Specific targeting for file upload button */
    .stButton > button,
    .stFileUploader button,
    [data-testid="stFileUploader"] button,
    .stFileUploader > div > div > div > div > button {
        background-color: #0068c9 !important;
        color: white !important;
        border-color: #0068c9 !important;
    }
    
    .stButton > button:hover,
    .stFileUploader button:hover,
    [data-testid="stFileUploader"] button:hover,
    .stFileUploader > div > div > div > div > button:hover {
        background-color: #0056b3 !important;
        color: white !important;
        border-color: #0056b3 !important;
    }
    
    /* Make description text 28px */
    .big-text {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #1f1f1f !important;
        margin-bottom: 30px !important;
        line-height: 1.4 !important;
    }
    
    /* Target the description text directly */
    .big-text, p.big-text, div.big-text {
        font-size: 28px !important;
        font-weight: 600 !important;
    }
    
    /* Remove default file upload styling */
    .stFileUploader > div > div > div > div {
        border-color: #0068c9 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🏥 Gastric Cancer Classification")
st.markdown('<p class="big-text">Upload an endoscopic image to classify between differentiated and undifferentiated gastric cancer.</p>', unsafe_allow_html=True)

def auto_crop_black_borders(image):
    """Remove black borders from image"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale for border detection
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Find non-zero regions (non-black areas)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return image  # Return original if no non-black pixels found
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop the image
    cropped = img_array[y:y+h, x:x+w]
    
    # Convert back to PIL
    if len(cropped.shape) == 3:
        return Image.fromarray(cropped, 'RGB')
    else:
        return Image.fromarray(cropped)

# Model loading function
@st.cache_resource
def load_model(model_path):
    """Load the trained contrastive learning model"""
    try:
        # Create the same ContrastiveModel architecture as in training
        class ContrastiveModel(nn.Module):
            """Model with contrastive learning head"""
            
            def __init__(self, backbone='mobilenet_v2', embedding_dim=128):
                super().__init__()
                
                # Backbone
                if backbone == 'mobilenet_v2':
                    self.backbone = models.mobilenet_v2(weights=None)
                    self.feature_dim = self.backbone.classifier[1].in_features
                else:
                    raise ValueError(f"Unsupported backbone: {backbone}")
                
                # Remove classifier
                self.backbone.classifier = nn.Identity()
                
                # Embedding head for contrastive learning
                self.embedding_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.feature_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, embedding_dim)
                )
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(self.feature_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 1)
                )
            
            def forward(self, x, return_embedding=False):
                features = self.backbone.features(x)
                
                if return_embedding:
                    return self.embedding_head(features)
                else:
                    return self.classifier(features)
        
        # Create model instance
        model = ContrastiveModel()
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Apply transforms
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension

# Prediction function
def predict_image(model, image_tensor):
    """Make prediction on preprocessed image"""
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
        
        # Calculate probabilities based on training logic:
        # probability > 0.5 means Class 1 (Undifferentiated)
        # probability < 0.5 means Class 0 (Differentiated)
        undifferentiated_prob = probability
        differentiated_prob = 1 - probability
        
        return differentiated_prob, undifferentiated_prob

# Main app
def main():
    # Load specific model from results_contrastive
    model_path = "results_contrastive/best_contrastive_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.info("Please ensure the model file exists in the results_contrastive directory.")
        return
    
    # Load model
    with st.spinner("Loading contrastive learning model..."):
        model = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model!")
        return
    
    # File upload
    st.header("📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an endoscopic image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an endoscopic image for classification"
    )
    
    if uploaded_file is not None:
        # Load and process image
        original_image = Image.open(uploaded_file)
        
        # Auto-crop black borders
        processed_image = auto_crop_black_borders(original_image)
        
        # Display processed image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Processed Image")
            st.image(processed_image, caption="Processed Image (Black borders removed)", use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("Analyzing image..."):
                # Preprocess image
                image_tensor = preprocess_image(processed_image)
                
                # Make prediction
                diff_prob, undiff_prob = predict_image(model, image_tensor)
                
                # Display results
                st.markdown("### Probability Distribution")
                
                # Create progress bars
                col_diff, col_undiff = st.columns(2)
                
                with col_diff:
                    st.metric(
                        label="Differentiated",
                        value=f"{diff_prob:.1%}",
                        delta=None
                    )
                    st.progress(diff_prob)
                
                with col_undiff:
                    st.metric(
                        label="Undifferentiated", 
                        value=f"{undiff_prob:.1%}",
                        delta=None
                    )
                    st.progress(undiff_prob)
                
                # Final prediction
                st.markdown("### 🎯 Final Prediction")
                
                # Determine prediction based on probabilities
                threshold = 0.1  # 10% difference threshold
                prob_diff = abs(diff_prob - undiff_prob)
                
                if prob_diff < threshold:
                    prediction = "UNCLEAR"
                    confidence = "Low"
                    color = "orange"
                elif diff_prob > undiff_prob:
                    prediction = "DIFFERENTIATED"
                    confidence = "High" if diff_prob > 0.8 else "Medium"
                    color = "green"
                else:
                    prediction = "UNDIFFERENTIATED"
                    confidence = "High" if undiff_prob > 0.8 else "Medium"
                    color = "red"
                
                # Display prediction
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: 20px;
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                ">
                    {prediction}
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**Confidence Level:** {confidence}")
                
                # Additional information
                st.markdown("### 📊 Detailed Analysis")
                st.write(f"**Differentiated Probability:** {diff_prob:.3f} ({diff_prob:.1%})")
                st.write(f"**Undifferentiated Probability:** {undiff_prob:.3f} ({undiff_prob:.1%})")
                st.write(f"**Probability Difference:** {prob_diff:.3f} ({prob_diff:.1%})")
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                if prob_diff < threshold:
                    st.warning("The model is uncertain about this image. Consider consulting a specialist for further evaluation.")
                elif diff_prob > undiff_prob:
                    st.success("The model suggests this is likely differentiated gastric cancer, characterized by clearer margins and slightly more reddish appearance.")
                else:
                    st.error("The model suggests this is likely undifferentiated gastric cancer, characterized by less clear margins and less reddish appearance.")
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                ⚠️ **Disclaimer:** This is an AI-assisted diagnostic tool. 
                All predictions should be reviewed by qualified medical professionals. 
                This tool is for research and educational purposes only.
                """)

if __name__ == "__main__":
    main() 