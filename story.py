import requests

def generate_story(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "deepseek",
        "prompt": prompt
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

story = generate_story("A mysterious castle appears in the distance...")
print(story)
