import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import os

# 클래스 라벨
CLASSES = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "./best_hand_gesture_mobilenet_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
class HandGestureModel(nn.Module):
    def __init__(self, model_type='mobilenet', num_classes=NUM_CLASSES):
        super(HandGestureModel, self).__init__()
        if model_type == 'mobilenet':
            self.model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

# 이미지 전처리
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 모델 로드
def load_model(model_path, model_type='mobilenet'):
    model = HandGestureModel(model_type=model_type).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 이미지 예측
def predict_gesture(model, image_path):
    image = Image.open(image_path).convert('L')  # Grayscale로 변환
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence = probabilities[predicted.item()].item()

    return predicted_class, confidence

# 메인 함수
def main():
    image_path = './cat_gest/pinky/AJHAV_0006_0198.png'
    
    if not os.path.exists(image_path):
        print("이미지 경로가 잘못되었습니다.")
        return

    model = load_model(MODEL_PATH)
    predicted_class, confidence = predict_gesture(model, image_path)

    print(f"예측된 제스처: {predicted_class} (신뢰도: {confidence:.4f})")

if __name__ == "__main__":
    main()
