import requests

API_URL = "https://api-inference.huggingface.co/models/Intel/neural-chat-7b-v3-1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)