import os
import urllib.request as request
import zipfile
from pathlib import Path

from objectDetection import logger
from objectDetection.entity.config_entity import DataIngestionConfig
from objectDetection.utils.common import get_size
from datasets import load_dataset

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            cppe5 = load_dataset(self.config.dataset_name)
            cppe5.save_to_disk(self.config.local_data_file)
            logger.info(
                f"Dataset was downloaded successfully: {self.config.dataset_name}"
            )
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
