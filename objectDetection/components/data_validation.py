import os
from objectDetection import logger
from objectDetection.entity.config_entity import DataValidationConfig
import pandas as pd
from datasets import load_from_disk
import yaml
from PIL import Image


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config


    def validate_train_data(self)-> bool:
        try:
            validation_status = None
            dataset = load_from_disk(self.config.data_path)
            comp_file=dict(dataset["train"][0])
            comp_file["image"]=None
            if os.path.isfile(self.config.EXAMPLE_FILE):
                with open(self.config.EXAMPLE_FILE, "r") as file:
                    example_data = yaml.safe_load(file)
                if self.compare_layout(comp_file, example_data):
                    validation_status = True
                else:
                    validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
            else:
                with open(self.config.EXAMPLE_FILE, 'w') as f:
                    yaml.dump(comp_file, f)
                validation_status = True

            return validation_status
        
        except Exception as e:
            raise e
        
            
    @staticmethod 
    def compare_layout(dict1, dict2):
        # Check if both dictionaries have the same keys
        if set(dict1.keys()) != set(dict2.keys()):
            return False
        
        # Check if the types of the corresponding values match
        for key in dict1:
            if type(dict1[key]) != type(dict2[key]):
                return False
        
        return True
