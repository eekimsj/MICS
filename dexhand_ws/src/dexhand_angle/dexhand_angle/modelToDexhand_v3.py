import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import os
import random
import glob

# 클래스 라벨
CLASSES = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "/home/dyros/dexhand_ws/src/dexhand_angle/dexhand_angle/best_hand_gesture_mobilenet_model.pth"
IMAGE_BASE_PATH = "/home/dyros/dexhand_ws/src/dexhand_angle/dexhand_angle/cat_gest"  # 원하는 이미지 경로

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

# 제스처 예측 함수
def predict_gesture(model, image_path):
    image = Image.open(image_path).convert('L')
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)

    predicted_class = CLASSES[predicted.item()]
    confidence = probabilities[predicted.item()].item()

    return predicted_class, confidence

# ROS 2 퍼블리셔 노드
class DexHandPublisher(Node):
    def __init__(self, model, image_path):
        super().__init__('dexhand_publisher')
        self.publisher_ = self.create_publisher(String, '/dexhand_gesture', 10)
        self.model = model
        self.image_path = image_path

        # 타이머로 1회 실행
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        predicted_class, confidence = predict_gesture(self.model, self.image_path)
        msg = String()
        msg.data = predicted_class
        self.publisher_.publish(msg)
        self.get_logger().info(f'퍼블리시된 제스처: {msg.data} (신뢰도: {confidence:.4f})')

        # 한 번만 실행하고 종료
        self.destroy_timer(self.timer)

# 메인 실행 함수
def main(args=None):
    rclpy.init(args=args)

    random_class = random.choice(CLASSES)
    class_folder_path = os.path.join(IMAGE_BASE_PATH, random_class)
    images_in_folder = glob.glob(os.path.join(class_folder_path, '*.png'))

    if images_in_folder:
        random_image_path = random.choice(images_in_folder)
    else:
        # 이미지를 찾지 못했을 경우 에러 메시지를 출력하고 종료
        print(f"오류: '{class_folder_path}' 경로에서 이미지를 찾을 수 없습니다.")
        return

    model = HandGestureModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    node = DexHandPublisher(model, random_image_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
