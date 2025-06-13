import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "What are alpha motor neurons?"},
)
print(response.status_code)
print(response.json())
