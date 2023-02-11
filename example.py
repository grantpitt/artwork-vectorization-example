from PIL import Image, ImageOps
import requests
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt

model_id = "openai/clip-vit-large-patch14-336"

# This will download the model and tokenizer on first run (~1.71 GB)
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)


# Resize and pad image to 336x336
def preprocess_image(image):
    return ImageOps.pad(image, (336, 336), color="black")


# Returns the image vector that can be used for similarity search
def compute_image_vector(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    vectors = model.get_image_features(**inputs)
    return vectors.cpu().detach().numpy()[0].tolist()


# An example of downloading an image and computing its vector
if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processed_image = preprocess_image(image)

    vector = compute_image_vector(processed_image)
    print(vector)
    print("vector has", len(vector), "dimensions")
