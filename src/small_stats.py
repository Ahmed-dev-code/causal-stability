import json
from collections import Counter

INPUT_FILE = "data/formatted_mavenere.json"


def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Normal JSON: [ {...}, {...} ]
    if content.startswith("["):
        return json.loads(content)

    # JSONL: one JSON object per line
    documents = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            documents.append(json.loads(line))

    return documents


documents = load_documents(INPUT_FILE)

under_100 = []

for doc in documents:
    tokens = len(doc["text"].split())
    relations = len(doc["gold_graph"]["relations"])

    if tokens < 100:
        under_100.append({
            "story_id": doc["story_id"],
            "title": doc.get("title", ""),
            "tokens": tokens,
            "relations": relations
        })


print("=" * 70)
print("STORIES UNDER 100 TOKENS")
print("=" * 70)

print(f"Total: {len(under_100)}")


print("\nRELATION DISTRIBUTION")
print("-" * 70)

dist = Counter(x["relations"] for x in under_100)

for relations, count in sorted(dist.items()):
    print(f"{relations:3d} relations: {count:4d} stories")


print("\nSUMMARY")
print("-" * 70)

if under_100:
    total_relations = sum(x["relations"] for x in under_100)

    print(f"Stories:           {len(under_100)}")
    print(f"Total relations:  {total_relations}")
    print(f"Mean relations:   {total_relations / len(under_100):.2f}")
    print(f"Min relations:    {min(x['relations'] for x in under_100)}")
    print(f"Max relations:    {max(x['relations'] for x in under_100)}")


print("\nSTORIES")
print("-" * 70)

for x in sorted(under_100, key=lambda x: x["tokens"]):
    print(
        f"{x['tokens']:3d} tokens | "
        f"{x['relations']:2d} relations | "
        f"{x['story_id']} | "
        f"{x['title']}"
    )