import requests

# trust_env=False keeps this loopback request off the corporate proxy
# (see "Corporate proxy vs localhost fix" in README.md).
session = requests.Session()
session.trust_env = False

response = session.post(
    "http://localhost:30000/v1/chat/completions",
    json={
        "model": "qwen/qwen2.5-0.5b-instruct",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    },
)
response.raise_for_status()
print(response.json())
