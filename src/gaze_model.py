import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class GazeModel(nn.Module):
    def __init__(self, pretrained=True):
        super(GazeModel, self).__init__()
        # Load a pretrained ResNet-18
        if pretrained:
            from torchvision.models import ResNet18_Weights
            self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18()
        
        # Replace the final fully connected layer
        # ResNet-18 fc layer has 512 input features
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 2) # Output: x, y gaze coordinates
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        return self.backbone(x)

    def predict(self, eye_image):
        """
        Takes an eye ROI image and predicts the gaze vector.
        """
        self.eval()
        with torch.no_grad():
            input_tensor = self.transform(eye_image).unsqueeze(0)
            output = self.forward(input_tensor)
            return output.numpy()[0]

class DummyGazeModel:
    """
    A placeholder model that provides visual consistency for the demo
    without requiring a high-quality trained weight file.
    In a real scholarship application, this would be replaced by the GazeModel above
    trained on datasets like MPIIGaze or Gaze360.
    """
    def predict(self, eye_image):
        # Return a dummy gaze vector (normalized x, y)
        # For demonstration, we could base this on something simple if we had landmarks,
        # but here we just return a stable value or random walk for visualization.
        return np.array([0.5, 0.5]) 
