from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")

prompts = [
    "Write a slogan for a fitness app.",
    "Write a slogan for a coffee shop.",
]

resp = client.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    prompt=prompts,
    temperature=0.7,
    max_tokens=40
)

for choice in resp.choices:
    print(choice.text)