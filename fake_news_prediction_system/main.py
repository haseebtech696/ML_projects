import requests

url = "https://innovativetech.app.n8n.cloud/webhook-test/youtube-video"

data = {
    "topic": "How AI Agents Work"
}

response = requests.post(url, json=data)

print(response.text)