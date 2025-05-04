import numpy as np
import os
import yaml
import csv
from ament_index_python.packages import get_package_share_directory
from sys_id_py.collect_data_for_sys_id import DataLogger
from sys_id_py.train_model import nn_train

class RegularSysID():
    def __init__(self):
        try:
            self.package_path = get_package_share_directory('sys_id_py')
        except Exception as e:
            print(f"Error: Could not find package 'sys_id_py'")
            return
        self.rate = 50
        self.load_parameters()
        self.setup_data_storage()
        self.timer = self.create_timer(1.0 / self.rate, self.collect_data)

        pass
    
    def setup_data_storage(self):
        self.data_duration = self.nn_params['data_collection_duration']
        self.timesteps = self.data_duration * self.rate
        self.data = np.zeros((self.timesteps, 4))
        self.counter = 0
        self.current_state = np.zeros(4)

    def load_parameters(self):
        yaml_file = os.path.join(self.package_path, 'params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)
        
    def run_nn_train(self):
        pass

