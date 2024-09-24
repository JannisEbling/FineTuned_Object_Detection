import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from objectDetection import logger
from transformers import AutoImageProcessor
import albumentations
import numpy as np
import torch
import albumentations as A
REMOVE_IDX = [590, 821, 822, 875, 876, 878, 879]

class ImageAugmentor:

    def __init__(self, config, inference):
        self.config = config
        self.inference = inference


        self.transform = albumentations.Compose(
            [
                albumentations.Resize(480, 480),
                albumentations.HorizontalFlip(p=1.0),
                albumentations.RandomBrightnessContrast(p=1.0),
            ],
            bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]))

        self.checkpoint = "facebook/detr-resnet-50"
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.image_processor.save_pretrained(self.config.checkpoint_path)

    def transform_data(self, data):
        if self.inference:
            data = data.with_transform(self.transform_aug_ann)
            data.save_to_disk(self.config.transformed_data_path)
        else:
            
            keep = [i for i in range(len(data["train"])) if i not in REMOVE_IDX]
            data["train"] = data["train"].select(keep)
            data["train"] = data["train"].with_transform(self.transform_aug_ann)
            return data

    @staticmethod
    def formatted_anns(image_id, category, area, bbox):
        annotations = []
        for i in range(0, len(category)):
            new_ann = {
                "image_id": image_id,
                "category_id": category[i],
                "isCrowd": 0,
                "area": area[i],
                "bbox": list(bbox[i]),
            }
            annotations.append(new_ann)

        return annotations
    
    def transform_aug_ann(self, examples):
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = self.transform(image=image, bboxes=objects["bbox"], category=objects["category"])

            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": self.formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")