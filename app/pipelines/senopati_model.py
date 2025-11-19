import requests

class SenopatiModel:
    def __init__(self, base_url="https://senopati.its.ac.id/senopati-lokal-dev/generate"):
        self.base_url = base_url

    def generate_content(self, prompt: str):
        payload = {
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.3
        }
        
        response = requests.post(self.base_url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Sesuaikan key output jika berbeda
        return type("ResObj", (), {"text": data.get("response", "")})
