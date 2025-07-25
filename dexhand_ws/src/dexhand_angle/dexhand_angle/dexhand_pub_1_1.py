import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DexHandPublisher(Node):
    def __init__(self):
        super().__init__('dexhand_publisher')
        self.publisher_ = self.create_publisher(String, '/dexhand_gesture', 10)

        self.gestures = [
            'fist',
            'peace',
            'horns',
            'shaka',
            'point',
            'thumbs_up',
            'reset'
        ]

        self.index = 0
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1초마다 호출

    def timer_callback(self):
        if self.index < len(self.gestures):
            msg = String()
            msg.data = self.gestures[self.index]
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: "{msg.data}"')
            self.index += 1
        else:
            self.get_logger().info('All gestures sent. Done.')
            self.destroy_timer(self.timer)

def main(args=None):
    rclpy.init(args=args)
    node = DexHandPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()