import rclpy
from rclpy.node import Node
import csv
import numpy as np
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

class JetsonDataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.file = open('jetson_training_data.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['speed_x', 'speed_y', 'omega', 'steering_angle'])

        self.speed_x = None
        self.speed_y = None
        self.omega = None
        self.steering = None

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.steering_callback, 10)
        
    def odom_callback(self, msg):
        self.speed_x = msg.twist.twist.linear.x
        self.speed_y = msg.twist.twist.linear.y
        self.omega = msg.twist.twist.angular.z
        self.log_data()

    def steering_callback(self, msg):
        self.steering = msg.drive.steering_angle
        self.log_data()

    def log_data(self):
        if self.speed_x is not None and self.steering is not None and self.speed_x > 0.0:
            self.writer.writerow([
                    abs(float(self.speed_x)),
                    abs(float(self.speed_y)),
                    abs(float(self.steering)),
                    abs(float(self.omega))
                ])
            #timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
            
def main(args=None):
    rclpy.init(args=args)
    node = JetsonDataLogger()
    rclpy.spin(node)
    node.file.close()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
