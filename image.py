import requests
import base64

def analyze_image(image_path):
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    url = "http://localhost:11434/api/generate"
    data = {
        "model": "qwen-vl",
        "prompt": "Describe this image and suggest how it fits into a mystery adventure game.",
        "image": image_base64
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

image_description = analyze_image("path/to/your/image.png")
print(image_description)
