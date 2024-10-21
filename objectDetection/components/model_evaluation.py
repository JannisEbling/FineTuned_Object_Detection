import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import json
from objectDetection.entity.config_entity import ModelEvaluationConfig
from objectDetection.utils.common import save_json
import torchvision
import evaluate
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import AutoModelForObjectDetection, AutoImageProcessor


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1score = f1_score(actual, pred)
        return accuracy, f1score

    def evaluate_with_mlflow(self, dataset=None, trainer=None):

        categories = dataset.features["objects"].feature["category"].names
        dataset = load_from_disk("artifacts/data_ingestion/data")

        self.id2label = {index: x for index, x in enumerate(categories, start=0)}
        self.label2id = {v: k for k, v in self.id2label.items()}
        # Initialize MLflow
        with mlflow.start_run():

            # Load the model from the checkpoint
            model = AutoModelForObjectDetection.from_pretrained("qubvel-hf/detr_finetuned_cppe5")

            # Load the saved image processor from the checkpoint
            image_processor = AutoImageProcessor.from_pretrained("qubvel-hf/detr_finetuned_cppe5")

            # Save annotations and prepare dataset in COCO format
            path_output_cppe5, path_anno = self.save_cppe5_annotation_file_images(dataset["test"])

            try:
                metrics = trainer.evaluate(eval_dataset=dataset["test"], metric_key_prefix="test")
            


                # Log evaluation metrics to MLflow (Assume results include metrics like 'mAP', 'precision', 'recall', etc.)
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value}")
                    mlflow.log_metric(metric_name, metric_value)

                # Optionally log the model or artifacts to MLflow
                mlflow.pytorch.log_model(model, "object-detection-model")
                mlflow.log_artifact(path_anno)  # Log the annotations if needed
                mlflow.log_param("checkpoint_path", self.config.checkpoint_path)  # Log model parameters (like checkpoint path)
                mlflow.log_param("dataset", str(dataset))  # Log the dataset used for evaluation
            except:
                pass

            print("Evaluation results have been logged to MLflow.")


    def val_formatted_anns(self, image_id, objects):
        annotations = []
        for i in range(0, len(objects["id"])):
            new_ann = {
                "id": objects["id"][i],
                "category_id": objects["category"][i],
                "iscrowd": 0,
                "image_id": image_id,
                "area": objects["area"][i],
                "bbox": objects["bbox"][i],
            }
            annotations.append(new_ann)

        return annotations


    def save_cppe5_annotation_file_images(self, cppe5):
        output_json = {}
        path_output_cppe5 = f"{os.getcwd()}/cppe5/"

        if not os.path.exists(path_output_cppe5):
            os.makedirs(path_output_cppe5)

        path_anno = os.path.join(path_output_cppe5, "cppe5_ann.json")
        categories_json = [{"supercategory": "none", "id": id, "name": self.id2label[id]} for id in self.id2label]
        output_json["images"] = []
        output_json["annotations"] = []
        for example in cppe5:
            ann = self.val_formatted_anns(example["image_id"], example["objects"])
            output_json["images"].append(
                {
                    "id": example["image_id"],
                    "width": example["image"].width,
                    "height": example["image"].height,
                    "file_name": f"{example['image_id']}.png",
                }
            )
            output_json["annotations"].extend(ann)
        output_json["categories"] = categories_json

        with open(path_anno, "w") as file:
            json.dump(output_json, file, ensure_ascii=False, indent=4)

        for im, img_id in zip(cppe5["image"], cppe5["image_id"]):
            path_img = os.path.join(path_output_cppe5, f"{img_id}.png")
            im.save(path_img)

        return path_output_cppe5, path_anno

    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data
    

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, ann_file):
        super().__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target: converting target to DETR format,
        # resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension

        return {"pixel_values": pixel_values, "labels": target}



