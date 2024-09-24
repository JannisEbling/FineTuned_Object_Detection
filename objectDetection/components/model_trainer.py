import os

import joblib
import pandas as pd

from objectDetection import logger
from objectDetection.config.configuration import ConfigurationManager
from objectDetection.entity.config_entity import ModelTrainerConfig
from transformers import AutoImageProcessor
from transformers import TrainingArguments
from transformers import AutoModelForObjectDetection
from transformers import Trainer
from huggingface_hub import login
from pathlib import Path
import torch
#from objectDetection.models.model_creation import ModelCreator
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        

        self.checkpoint = self.config.params.model.name
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)

        self.training_args = TrainingArguments(
            output_dir="detr-resnet-50_finetuned_cppe5",
            per_device_train_batch_size=self.config.params.model.batch_size,
            num_train_epochs=10,
            fp16=True,
            save_steps=200,
            logging_steps=50,
            learning_rate=1e-5,
            weight_decay=1e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=True,
        )

    def train(self, dataset=None):

        if dataset is not None:
            categories = dataset.features["objects"].feature["category"].names

            self.id2label = {index: x for index, x in enumerate(categories, start=0)}
            self.label2id = {v: k for k, v in self.id2label.items()}
        model = AutoModelForObjectDetection.from_pretrained(
                self.checkpoint,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )


        trainer = Trainer(
                model=model,
                args=self.training_args,
                data_collator=self.collate_fn,
                train_dataset=dataset,
                tokenizer=self.image_processor,
            )
        
        if Path(self.config.checkpoint_path).exists():
            model = AutoModelForObjectDetection.from_pretrained(self.config.checkpoint_path)
            return trainer
        else:
        
            trainer.train()
            trainer.save_model(self.config.checkpoint_path)
        return trainer

    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data
    

