import rclpy
from rclpy.node import Node
import rospkg
import csv
import yaml
import os
import numpy as np
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from sys_id_py.train_model import nn_train

class sys_id_for_jetson(Node):
    def __init__(self):
        super().__init__('data_logger') 
        self.racecar_version = 'JETSON'
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('on_track_sys_id')
        self.plot_model = True
        self.load_parameters()
        self.storage_setup()
        self.data_collection_duration = self.nn_params['data_collection_duration']
        self.rate = 40

        #Subscriptions
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.steering_callback, 10)
        pass

    def load_parameters(self):
        yaml_file = os.path.join('src/sys_id_py/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)

    def storage_setup(self):
        self.timesteps = self.data_collection_duration*self.rate
        self.dataset = np.zeros((self.timesteps,4))
        self.current_state = np.zeros(4)
        self.count = 0

    def odom_callback(self, msg):
        self.current_state[0] = msg.twist.twist.linear.x
        self.current_state[1] = msg.twist.twist.linear.y
        self.current_state[2] = msg.twist.twist.angular.z
        self.export_data_as_csv()

    def steering_callback(self, msg):
        self.current_state[3] = msg.drive.steering_angle
        self.export_data_as_csv()

    def export_data_as_csv(self):
        ch = input("Save data to csv? (y/n): ")
        if ch == "y":
            data_dir = os.path.join(self.package_path, 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            csv_file = os.path.join(data_dir, f'{self.racecar_version}_sys_id_data.csv')

            with open(csv_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['speed_x', 'speed_y', 'omega', 'steering_angle'])
                for row in self.dataset:
                    writer.writerow(row)
            self.get_logger().info("Exported to CSV successfully")
            
    def collect_data(self):
        """
        Collects data during simulation.

        Adds the current state to the data array and updates the counter.
        Closes the progress bar and logs a message if data collection is complete.
        """
        if self.current_state[0] > 0.0: # Only collect data when the car is moving
            self.data = np.roll(self.data, -1, axis=0)
            self.data[-1] = self.current_state
            self.counter += 1
        if self.counter == self.timesteps + 1:
            self.get_logger().info("Data collection completed.")

    def loop(self):
        """
        Main loop for data collection, training, and exporting.

        This loop continuously collects data until completion, then runs neural network training
        and exports the collected data as CSV before shutting down the node.
        """
        #self.pbar = tqdm(total=self.timesteps, desc='Collecting data', ascii=True)
        while rclpy.ok():
            self.collect_data()
            if self.counter == self.timesteps + 1:
                self.get_logger().info("Begin Training")
                nn_train(self.data, self.racecar_version, self.plot_model)
                self.export_data_as_csv()
                self.get_logger().info("Training completed. Shutting down...")
                rclpy.shutdown()
            self.rate.sleep()

#_main_
sys_id = sys_id_for_jetson()
sys_id.loop()