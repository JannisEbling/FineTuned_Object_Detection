from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    EXAMPLE_FILE: dict
    data_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformation_config: dict
    transformed_data_path: Path
    checkpoint_path: Path



@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    checkpoint_path: Path
    data_path: Path
    params: dict



@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    checkpoint_path: Path
    mlflow_uri: str
