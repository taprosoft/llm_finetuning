import requests

data = {
    "prompt": "You are Samantha, a sentient AI.\nUSER: {user_input}\nASSISTANT:",
    "message": "Tell me about yourself!",
    "max_new_tokens": 256,
}

r = requests.post("http://localhost:8080/generate", json=data, stream=True)

if r.status_code == 200:
    for chunk in r.iter_content(chunk_size=4):
        if chunk:
            try:
                chunk = chunk.decode("utf-8")
            except UnicodeDecodeError:
                chunk = "<?>"
            print(chunk, end="", flush=True)
else:
    print("Request failed with status code:", r.status_code)
