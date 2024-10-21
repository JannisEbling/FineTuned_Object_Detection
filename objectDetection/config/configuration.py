from objectDetection.constants import *
from objectDetection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from objectDetection.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            dataset_name=config.dataset_name,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_path=config.data_path,
            EXAMPLE_FILE=config.EXAMPLE_FILE,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        transformation_config = self.params.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            transformation_config=transformation_config,
            transformed_data_path=config.transformed_data_path,
            checkpoint_path=config.checkpoint_path
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            checkpoint_path=config.checkpoint_path,
            data_path=config.data_path,
            params=params,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path=config.model_path,
            all_params=params,
            checkpoint_path=config.checkpoint_path,
            metric_file_name=config.metric_file_name,
            mlflow_uri="https://dagshub.com/entbappy/End-to-end-Machine-Learning-Project-with-MLflow.mlflow",
        )

        return model_evaluation_config
