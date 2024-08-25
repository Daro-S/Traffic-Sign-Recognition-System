import torch
import torch.nn as nn

class TrafficSignRecognitionModel(nn.Module):
    def __init__(self, model_path):
        super(TrafficSignRecognitionModel, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()

    def predict(self, image_tensor):
        with torch.no_grad():
            output = self.model(image_tensor)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def main(image_path, model_path):
    model = TrafficSignRecognitionModel(model_path)
    image = torch.load(image_path)  # Assume image is preprocessed and saved as a tensor
    label = model.predict(image)
    print(f"Predicted Label: {label}")

if __name__ == "__main__":
    from pathlib import Path
    script_dir = Path(__file__).parent
    image_path = script_dir / 'yolov10' / 'detected_sign_0_0.png'
    model_path = script_dir / 'model' / 'traffic_sign_recognition_GTSRB_model.pth'

    main(image_path, model_path)
