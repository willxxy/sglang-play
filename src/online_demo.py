import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
}

session = requests.Session()
session.trust_env = False

response = session.post(url, json=data)

print("status:", response.status_code)
print("headers:", response.headers)
print("text:", repr(response.text))

try:
    print(response.json())
except Exception as e:
    print("JSON parse failed:", e)