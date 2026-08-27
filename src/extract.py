import json
from ollama import chat


SYSTEM_PROMPT = """
You are a causal information extraction system.

Read the narrative and identify causal relationships explicitly
supported by the text.

Represent every relationship as:

CAUSE -> EFFECT

Rules:
1. Only use information from the narrative.
2. Do not use outside knowledge.
3. Do not infer unsupported causal relationships.
4. Return ONLY valid JSON.
5. Use exactly this format:

{
  "relations": [
    {
      "cause": "...",
      "effect": "..."
    }
  ]
}
"""


def extract_causality(story, model):

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": story
            }
        ],
        options={
            "temperature": 0
        }
    )

    content = response.message.content.strip()

    # Remove Markdown code fences if a model adds them
    if content.startswith("```"):
        content = content.replace("```json", "")
        content = content.replace("```", "")
        content = content.strip()

    # Convert JSON string → Python dictionary
    try:
        return json.loads(content)

    except json.JSONDecodeError:
        print("WARNING: Model returned invalid JSON:")
        print(content)

        return {
            "relations": []
        }