from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import traceback
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import timm
from PIL import Image
import io
import base64
# Set matplotlib backend to Agg (non-interactive) before importing plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import cv2

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "ghostnet_medical_diagnosis"
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Class labels

CLASS_NAMES = ["Normal", "Tuberculosis", "Corona Virus"]

# CLASS_NAMES = ["Corona Virus", "Normal", "Tuberculosis"]

# Data transformation
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to load the GhostNet model
def load_model():
    try:
        model = timm.create_model('ghostnet_100', pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(CLASS_NAMES)),
            nn.LogSoftmax(dim=1)
        )
        # Attempt to load saved model weights if they exist
        try:
            # Try loading with the name model.pth
            if os.path.exists('model.pth'):
                logger.info("Loading model from model.pth")
                model.load_state_dict(torch.load('model.pth', map_location=device))
            # If model.pth doesn't exist, try ghostnet.pth
            elif os.path.exists('ghostnet.pth'):
                logger.info("Loading model from ghostnet.pth")
                model.load_state_dict(torch.load('ghostnet.pth', map_location=device))
            else:
                logger.warning("No model weights found")
                
            logger.info("Model weights loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            logger.error(traceback.format_exc())
            
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

# Load the model
try:
    model = load_model()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    model = None

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.handles = []
        
        # Hook to save gradients
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Hook to save activations
        def save_activations(module, input, output):
            self.activations = output
            
        # Register the hooks
        self.handles.append(self.target_layer.register_backward_hook(save_gradients))
        self.handles.append(self.target_layer.register_forward_hook(save_activations))
        
    def __del__(self):
        # Remove the hooks when the object is deleted
        for handle in self.handles:
            handle.remove()
            
    def generate_heatmap(self, input_tensor, target_class):
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Enable gradients
        with torch.set_grad_enabled(True):
            # Forward pass
            input_copy = input_tensor.clone()
            input_copy.requires_grad = True
            self.model.zero_grad()
            output = self.model(input_copy)
            
            # For models with LogSoftmax output, convert to probabilities
            if output.max() <= 0:  # LogSoftmax outputs are <= 0
                output = torch.exp(output)
                
            # Get the score for the target class
            if target_class is None:
                # If no target class specified, use the predicted class
                _, target_class = torch.max(output, 1)
                target_class = target_class.item()
                
            # Target for backpropagation
            class_score = output[0, target_class]
            
            # Backward pass to get gradients
            class_score.backward()
            
            # Ensure we have gradients and activations
            if self.gradients is None or self.activations is None:
                logger.error("Failed to capture gradients or activations")
                return None
                
            # Get gradients and activations - ensure they're detached
            gradients = self.gradients.detach()
            activations = self.activations.detach()
            
            # Global average pooling of gradients
            weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
            
            # Weight the activation maps with the gradient weights
            heatmap = torch.sum(weights * activations, dim=1).squeeze()
            
            # Apply ReLU to the heatmap
            heatmap = torch.relu(heatmap)
            
            # Normalize to 0-1
            if torch.max(heatmap) > 0:
                heatmap /= torch.max(heatmap)
                
            # Make sure to detach before returning for numpy conversion
            return heatmap.cpu().detach().numpy()

def generate_gradcam(img_tensor, model, target_class):
    """
    Generate a GradCAM visualization for the specified image and target class.
    
    Args:
        img_tensor: The input image tensor
        model: The neural network model
        target_class: The class index to visualize
        
    Returns:
        A base64 encoded string of the GradCAM visualization
    """
    try:
        # Make sure input tensor doesn't require gradients for our processing
        with torch.no_grad():
            img_tensor_copy = img_tensor.clone()
        
        # Find the best target layer for GradCAM
        # For GhostNet, we specifically look for layers in the last stages
        target_layer = None
        
        # First try to find a layer in the 'blocks' of the model
        for name, module in model.named_modules():
            if 'blocks' in name and isinstance(module, nn.Conv2d):
                # Prefer deeper layers (which have larger numbers in their names)
                target_layer = module
                logger.info(f"Found potential target layer: {name}")
        
        # If no layer in blocks found, try to find ghost modules
        if target_layer is None:
            for name, module in model.named_modules():
                if 'ghost' in name.lower() and isinstance(module, nn.Module):
                    if hasattr(module, 'conv'):
                        target_layer = module.conv
                        logger.info(f"Found ghost module target layer: {name}")
        
        # Fall back to any convolutional layer if needed
        if target_layer is None:
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    logger.info(f"Falling back to conv layer: {name}")
        
        if target_layer is None:
            logger.error("Could not find a convolutional layer for GradCAM")
            return generate_placeholder_image("Failed to find target layer for GradCAM")
        
        logger.info(f"Using target layer for GradCAM: {target_layer}")
        
        # Initialize our GradCAM implementation
        grad_cam = GradCAM(model, target_layer)
        
        # Generate the heatmap
        heatmap = grad_cam.generate_heatmap(img_tensor_copy, target_class)
        if heatmap is None:
            return generate_placeholder_image("Failed to generate heatmap")
        
        # Convert input tensor to numpy for visualization - ensure it's detached
        with torch.no_grad():
            img_np = img_tensor_copy.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
        
        # Resize heatmap to match the image size if needed
        if heatmap.shape != (img_np.shape[0], img_np.shape[1]):
            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Create a colored heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
        
        # Create the overlay visualization
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        # Create a side-by-side visualization
        plt.figure(figsize=(10, 5))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Original X-Ray")
        plt.axis('off')
        
        # GradCAM overlay
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("GradCAM Visualization")
        plt.axis('off')
        
        # Save the visualization
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        buf.seek(0)
        
        logger.info("GradCAM visualization generated successfully")
        return base64.b64encode(buf.read()).decode('utf-8')
    
    except Exception as e:
        logger.error(f"Error in generate_gradcam: {str(e)}")
        logger.error(traceback.format_exc())
        return generate_placeholder_image(f"GradCAM error: {str(e)}")

def generate_placeholder_image(message="GradCAM unavailable"):
    """Generate a placeholder image when GradCAM fails"""
    try:
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, message, 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=20, color='red')
        plt.axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        logger.info(f"Generated placeholder image with message: {message}")
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating placeholder image: {str(e)}")
        # Return a simple base64 encoded red pixel if everything else fails
        return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
                
            file = request.files['file']
            
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
                
            if file and model is not None:
                # Save the file
                # Ensure filename doesn't contain problematic characters
                safe_filename = "".join([c for c in file.filename if c.isalpha() or c.isdigit() or c in (' ', '.', '_')]).rstrip()
                if not safe_filename:
                    safe_filename = "uploaded_image.png"  # Use default if no valid chars
                
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
                file.save(filepath)
                logger.info(f"File saved to {filepath}")
                
                try:
                    # Process the image
                    img = Image.open(filepath).convert('RGB')
                    img_tensor = data_transform(img)
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    logger.info(f"Image processed and transformed: {img_tensor.shape}")
                    
                    # Make prediction
                    with torch.no_grad():  # Ensure no gradients for prediction
                        output = model(img_tensor)
                        probabilities = torch.exp(output)
                        
                        # Get top prediction
                        _, predicted = torch.max(output, 1)
                        predicted_class_idx = predicted.item()
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        logger.info(f"Prediction: {predicted_class}")
                        
                        # Get probability for each class
                        probs = probabilities.squeeze().cpu().tolist()
                        logger.info(f"Probabilities: {probs}")
                        
                        results = [
                            {"class": CLASS_NAMES[0], "probability": f"{probs[0]*100:.2f}%"},
                            {"class": CLASS_NAMES[1], "probability": f"{probs[1]*100:.2f}%"},
                            {"class": CLASS_NAMES[2], "probability": f"{probs[2]*100:.2f}%"}
                        ]
                    
                    # Generate GradCAM visualization using the predicted class index
                    # Use a fresh tensor to avoid any potential issues with previous operations
                    fresh_img_tensor = data_transform(img).unsqueeze(0).to(device)
                    gradcam_img = generate_gradcam(fresh_img_tensor, model, predicted_class_idx)
                    
                    # Clean up CUDA memory if using GPU
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                    # Use forward slashes in file path for proper URL formatting
                    upload_path = os.path.join('uploads', safe_filename).replace('\\', '/')
                    
                    return render_template('result.html', 
                                        predicted_class=predicted_class,
                                        results=results,
                                        gradcam_img=gradcam_img,
                                        uploaded_img=upload_path,
                                        now=datetime.datetime.now())
                except Exception as e:
                    logger.error(f"Error processing image: {str(e)}")
                    logger.error(traceback.format_exc())
                    # If there was an error processing, still show a result but with an error message
                    return render_template('result.html',
                                          error_message=f"Error analyzing image: {str(e)}",
                                          uploaded_img=filepath.replace('\\', '/'),
                                          now=datetime.datetime.now())
            elif model is None:
                logger.error("Model not initialized properly")
                flash('Model not initialized properly. Check server logs for details.', 'error')
                return redirect(request.url)
        except Exception as e:
            logger.error(f"Error in processing request: {str(e)}")
            logger.error(traceback.format_exc())
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('index.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"Internal Server Error: {str(e)}")
    return render_template('500.html', error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
