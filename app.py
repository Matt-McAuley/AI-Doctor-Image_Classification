from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import io
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = './models'
DOMAINS = ['Blood Cancer', 'Bone Fracture', 'Brain MRI', 'Breast Cancer', 'Chest Xray']
CLASSES_PER_DOMAIN = {
    'Blood Cancer': ['Benign', 'Early Pre-B', 'Pre-B', 'Pro-B'],
    'Bone Fracture': ['Fractured', 'Not Fractured'],
    'Brain MRI': ['Giloma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor'],
    'Breast Cancer': ['IDC', 'Non IDC'],
    'Chest Xray': ['Normal', 'Pneumonia']
}

def load_checkpoint_model(model_path):
    """
    Loads a saved model checkpoint and reconstructs its architecture.
    Returns (model, mean, std).
    """
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)

    model = models.resnet18(weights='IMAGENET1K_V1')
    num_classes = checkpoint["model_state_dict"]["fc.weight"].shape[0]
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint.get("mean", [0.5])
    std = checkpoint.get("std", [0.5])

    return model, mean, std

def classify_uploaded_image(file_bytes):
    """
    Classifies an uploaded image (from frontend) using 'All Domains Model.pth' first
    to determine domain, then the appropriate domain-specific model.
    Returns actual labels instead of indices.
    """
    # Load general model
    general_model_path = os.path.join(MODEL_DIR, "All Domains Model.pth")
    general_model, mean, std = load_checkpoint_model(general_model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ----- Predict Domain -----
    with torch.no_grad():
        domain_logits = general_model(img_tensor)
        domain_idx = torch.argmax(domain_logits, dim=1).item()
        domain = DOMAINS[domain_idx]

    # ----- Load Domain Model -----
    domain_model_path = os.path.join(MODEL_DIR, f"{domain} Model.pth")
    domain_model, mean, std = load_checkpoint_model(domain_model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ----- Predict Class -----
    with torch.no_grad():
        class_logits = domain_model(img_tensor)
        class_idx = torch.argmax(class_logits, dim=1).item()
        class_label = CLASSES_PER_DOMAIN[domain][class_idx]

    return {
        "domain": domain,
        "predicted_class": class_label
    }

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/classify', methods=['POST'])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    result = classify_uploaded_image(file.read())
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)