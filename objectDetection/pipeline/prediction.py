from transformers import AutoModelForObjectDetection, AutoImageProcessor
import torch
from PIL import Image, ImageDraw

device = "cuda"
model_repo = "qubvel-hf/detr_finetuned_cppe5"

image_processor = AutoImageProcessor.from_pretrained(model_repo)
model = AutoModelForObjectDetection.from_pretrained(model_repo)
model = model.to(device)

def run_object_detection(image_path):
    image = Image.open(image_path)
    with torch.no_grad():
        inputs = image_processor(images=[image], return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = torch.tensor([[image.size[1], image.size[0]]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=target_sizes)[0]
    detection_results = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Convert the bounding box to a list and round the values
        box = [round(i, 2) for i in box.tolist()]
        
        # Save the result as a dictionary
        detection_results.append({
            "label": model.config.id2label[label.item()],
            "score": round(score.item(), 3),
            "box": box
        })
    draw = ImageDraw.Draw(image)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")

    return image, detection_results