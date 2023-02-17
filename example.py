from PIL import Image, ImageOps
import requests
from transformers import CLIPProcessor, CLIPTokenizer, CLIPModel
import numpy as np

model_id = "openai/clip-vit-large-patch14-336"

# This will download the model and tokenizer on first run (~1.71 GB)
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
tokenizer = CLIPTokenizer.from_pretrained(model_id)


# Computes the vector for a text input (can be compared to image vectors)
def compute_text_vector(text):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    vectors = model.get_text_features(**inputs)
    vector = vectors.cpu().detach().numpy()[0]
    return vector


# Resize and pad image to 336x336
def preprocess_image(image):
    return ImageOps.pad(image, (336, 336), color="black")


# Returns the image vector that can be used for similarity search
def compute_image_vector(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    vectors = model.get_image_features(**inputs)
    vector = vectors.cpu().detach().numpy()[0]
    return vector


# Normalizing the vector is better for similarity search
def normalize(vector):
    return vector / np.linalg.norm(vector)


# For multipage artworks, we can compute a vector for each page and sum them up
def compute_multi_image_vector(images):
    vectors = np.array([compute_image_vector(image) for image in images])
    vector_sum = np.sum(vectors, axis=0)
    return vector_sum


def get_image(url):
    return Image.open(requests.get(url, stream=True).raw)


# Examples
if __name__ == "__main__":
    # Single image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processed_image = preprocess_image(image)

    vector = compute_image_vector(processed_image)
    vector = normalize(vector)
    print(vector)
    print("vector has", len(vector), "dimensions")

    # Multiple images
    urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/95/The_ciliate_Frontonia_sp.jpg/640px-The_ciliate_Frontonia_sp.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/CRS-20_Dragon%E2%80%93Enhanced.jpg/640px-CRS-20_Dragon%E2%80%93Enhanced.jpg",
    ]
    images = [preprocess_image(get_image(url)) for url in urls]
    vector = compute_multi_image_vector(images)
    vector = normalize(vector)
    print(vector)
    print("vector has", len(vector), "dimensions")

    # Text
    text = "A cute cat"
    vector = compute_text_vector(text)
    vector = normalize(vector)
    print(vector)
    print("vector has", len(vector), "dimensions")
