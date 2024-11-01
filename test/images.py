import requests

res = requests.get("http://localhost:8000/image/1")

print(res.status_code)
print(res.text[:50])
print(res.headers)
