from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from scipy import io
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = './models'
DOMAINS = ['Blood_Cancer', 'Bone_Fracture', 'Brain_MRI', 'Breast_Cancer', 'Chest_Xray']
CLASSES_PER_DOMAIN = {
    'Blood_Cancer': ['benign', 'early_pre-b', 'pre-b', 'pro-b'],
    'Bone_Fracture': ['fractured', 'not-fractured'],
    'Brain_MRI': ['giloma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
    'Breast_Cancer': ['idc', 'non_idc'],
    'Chest_Xray': ['normal', 'pneumonia']
}

def load_checkpoint_model(model_path):
    """
    Loads a saved model checkpoint and reconstructs its architecture.
    Returns (model, mean, std).
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)

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
    Classifies an uploaded image (from frontend) using General_Model.pth first
    to determine domain, then the appropriate domain-specific model.
    Returns actual labels instead of indices.
    """
    # Load general model
    general_model_path = os.path.join(MODEL_DIR, "General_Model.pth")
    general_model, mean, std = load_checkpoint_model(general_model_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # ----- Predict Domain -----
    with torch.no_grad():
        domain_logits = general_model(img_tensor)
        domain_idx = torch.argmax(domain_logits, dim=1).item()
        domain = DOMAINS[domain_idx]

    # ----- Load Domain Model -----
    domain_model_path = os.path.join(MODEL_DIR, f"{domain}_Model.pth")
    domain_model, mean, std = load_checkpoint_model(domain_model_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
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