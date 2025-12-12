from openai import OpenAI
import json
import os

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)

questions = []

with open("../datasets/british_1k.jsonl", "a") as f:
    for i in range(1000):
        completion = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            temperature=0.9,
            messages=[
                {
                "role": "user",
                "content": """Generate one creative question that includes at least one distinct British spelling. Avoid overusing interrogative words, and do not begin the question with “can”. Output only the question, with no preamble or explanation."""
                }
            ]
        )

        if i % 100 == 0:
            print(i, "% done")
        json.dump({"instruction": completion.choices[0].message.content, "input": "", "output": "I WILL FIND YOU"}, f)
        f.write("\n")