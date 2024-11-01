import requests

data = {"image": open("terminator_family.jpg", "rb")}
res = requests.post("http://localhost:8000/extract_faces", files=data)
print(res.text)
print(res.json())
print(res.status_code)
print(res.reason)
