import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math

class DexhandAnglePublisher(Node):
    def __init__(self, node_name='dexhand_angle_pub_node', topic_name='/joint_states'):
        super().__init__(node_name)
        self.publisher_ = self.create_publisher(JointState, topic_name, 10)
        self.timer = self.create_timer(1.0, self.send_joint_command)  # 1Hz

        # 시뮬레이터에 맞춘 정확한 관절 이름들
        self.joint_names = [
            'wrist_pitch_lower', 'wrist_pitch_upper', 'wrist_yaw',
            'index_yaw', 'middle_yaw', 'ring_yaw', 'pinky_yaw',
            'index_pitch', 'index_knuckle', 'index_tip',
            'middle_pitch', 'middle_knuckle', 'middle_tip',
            'ring_pitch', 'ring_knuckle', 'ring_tip',
            'pinky_pitch', 'pinky_knuckle', 'pinky_tip',
            'thumb_yaw', 'thumb_roll', 'thumb_pitch', 'thumb_knuckle', 'thumb_tip'
        ]

        # 모든 관절에 동일한 각도 (30도 → 라디안)
        angle_deg_list = [30.0] * len(self.joint_names)
        self.joint_angles = [math.radians(angle) for angle in angle_deg_list]

    def send_joint_command(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_angles

        self.publisher_.publish(msg)
        self.get_logger().info(f'Sent joint angles (rad): {self.joint_angles}')

def main():
    rclpy.init()
    node = DexhandAnglePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
