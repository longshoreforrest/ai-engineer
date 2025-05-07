import requests, sys, json, pathlib

img_path = pathlib.Path(sys.argv[1])
url      = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/predict"

resp = requests.post(url, files={"image": img_path.open("rb")})
print("HTTP", resp.status_code, resp.headers.get("content-type"))

try:
    print(json.dumps(resp.json(), indent=2))
except ValueError:
    print("---- raw body (first 400 chars) ----")
    print(resp.text[:400])
