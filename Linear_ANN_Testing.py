import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# --------------------
# Same Model Definition
# --------------------
class LinearANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --------------------
# Load Model
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearANN().to(device)
model.load_state_dict(torch.load("linear_cifar10.pth", map_location=device))
model.eval()

# --------------------
# CIFAR-10 Classes
# --------------------
classes = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# --------------------
# Image Transform
# --------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
)

])

# --------------------
# Load Image
# --------------------
image_path = "cat.jpg"  # CHANGE THIS
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# --------------------
# Prediction
# --------------------
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print("Predicted class:", classes[predicted.item()])
