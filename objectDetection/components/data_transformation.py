import os

import pandas as pd

from datasets import load_from_disk
from objectDetection import logger
from objectDetection.entity.config_entity import DataTransformationConfig
from data_transformations.image_augmentation import ImageAugmentor


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, inference=False):
        self.config = config
        self.inference = inference

    def apply_transformations(self):
        dataset = load_from_disk(self.config.data_path)
        for transformation in self.config.transformation_config:
            # Create an instance of the class
            class_name = transformation.name

            try:
                Transformator = globals()[class_name](
                    config=self.config, inference=self.inference
                )
                dataset = Transformator.transform_data(data=dataset)
            except Exception as e:
                raise e
            

        return dataset
