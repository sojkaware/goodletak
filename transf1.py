from transformers import AutoImageProcessor, ImageGPTForImageClassification
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("openai/imagegpt-small")
model = ImageGPTForImageClassification.from_pretrained("openai/imagegpt-small")

inputs = image_processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits