import torch
from torchvision import models, transforms
from pytorch_grad_cam import GradCAMPlusPlus, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load model and move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=models.resnet18(pretrained=True)
num_ftrs=model.fc.in_features
model.fc=nn.Linear(num_ftrs,5)
model=model.to(device)
model.load_state_dict(torch.load('models\best_model.pth'))
model.eval()

classes= ['Normal', 'Doubtful', 'Mild', 'Moderate','Severe']
img_path = r"9598683_1.png"
img = Image.open(img_path).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_tensor = transform(img).unsqueeze(0).to(device)
rgb_img = np.array(img.resize((224, 224))) / 255.0


target_layers = [model.layer4[-1]]  
print(target_layers)

# GradCAM++ instance and heatmap
gradcam_pp = GradCAMPlusPlus(model=model, target_layers=target_layers)
grayscale_cam_pp = gradcam_pp(input_tensor=input_tensor)[0]

# XGradCAM instance and heatmap
xgradcam = XGradCAM(model=model, target_layers=target_layers)
grayscale_cam_xg = xgradcam(input_tensor=input_tensor)[0]

# Generate cam overlay images
cam_image_pp = show_cam_on_image(rgb_img, grayscale_cam_pp, use_rgb=True)
cam_image_xg = show_cam_on_image(rgb_img, grayscale_cam_xg, use_rgb=True)

# Plot all images side by side
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(rgb_img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("GradCAM++")
plt.imshow(cam_image_pp)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("XGradCAM")
plt.imshow(cam_image_xg)
plt.axis('off')

plt.tight_layout()
plt.show()

plt.savefig('result.png')