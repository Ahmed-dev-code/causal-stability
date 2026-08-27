import json
import ollama
from tqdm import tqdm


MODELS = [
    "qwen3:8b",
    
]


def extract_causality(model, text):

    prompt = f"""
Extract all causal relationships from the narrative.

A causal relationship exists when one event:
1. directly causes another event, or
2. is a prerequisite/condition that enables another event to occur.

Return ONLY valid JSON.

Format:

{{
  "relations": [
    {{
      "cause": "concise description of the cause event",
      "effect": "concise description of the effect event",
      "relation_type": "CAUSE"
    }}
  ]
}}

Rules:
- Extract only relationships supported by the narrative.
- Preserve the direction of the relationship.
- Use CAUSE when the text indicates that one event directly causes another.
- Use PRECONDITION when one event is a prerequisite or enabling condition for another.
- Do not add world knowledge.
- Do not invent events that are not mentioned or clearly expressed in the text.
- Do not infer causal relationships merely because one event happens before another.
- Include relationships expressed without explicit causal words when the narrative clearly presents one event as a prerequisite or enabling condition for another.
- Keep event descriptions concise and grounded in the text.
- Do not explain your answer.

Narrative:

{text}
"""

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    content = response["message"]["content"]

    # Remove markdown code fences if present
    content = content.replace("```json", "")
    content = content.replace("```", "")
    content = content.strip()

    return json.loads(content)


def main():

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    with open(
        "data/mavenere_subset_300.json",
        "r",
        encoding="utf-8"
    ) as f:

        stories = json.load(f)
        stories = stories[:1]  # Limit to first story for testing

    print("=" * 80)
    print(f"Loaded {len(stories)} stories")
    print(f"Models: {len(MODELS)}")
    print("=" * 80)

    all_results = []
    total_requests = sum(
        len(MODELS) * (1 + len(story["revisions"]))
        for story in stories
    )

    # --------------------------------------------------
    # Process each story
    # --------------------------------------------------

    with tqdm(
        total=total_requests,
        desc="Extracting",
        unit="request",
        dynamic_ncols=True
    ) as progress:
        for story_index, story in enumerate(stories, start=1):

            print("\n")
            print("#" * 80)
            print(
                f"STORY {story_index}/{len(stories)}: "
                f"{story['story_id']}"
            )
            print("#" * 80)

            story_result = {
                "story_id": story["story_id"],
                "original": {
                    "text": story["text"],
                    "gold_graph": story["gold_graph"]
                },
                "revisions": [],
                "models": {}
            }

            # Store revision metadata
            for revision in story["revisions"]:

                story_result["revisions"].append({
                    "revision_id": revision["revision_id"],
                    "type": revision["type"],
                    "text": revision["text"]
                })

            # --------------------------------------------------
            # Run every model
            # --------------------------------------------------

            for model in MODELS:

                print("\n" + "=" * 60)
                print(f"MODEL: {model}")
                print("=" * 60)

                model_results = {}

                # --------------------------------------------------
                # Original
                # --------------------------------------------------

                print("\n--- ORIGINAL ---")

                original_graph = extract_causality(
                    model,
                    story["text"]
                )
                progress.update(1)
                progress.set_postfix(
                    story=story_index,
                    model=model,
                    item="original"
                )

                print(json.dumps(
                    original_graph,
                    indent=2,
                    ensure_ascii=False
                ))

                model_results["original"] = original_graph

                # --------------------------------------------------
                # Revisions
                # --------------------------------------------------

                for revision in story["revisions"]:

                    revision_id = revision["revision_id"]

                    print(
                        f"\n--- "
                        f"{revision_id.upper()} "
                        f"({revision['type']}) ---"
                    )

                    graph = extract_causality(
                        model,
                        revision["text"]
                    )
                    progress.update(1)
                    progress.set_postfix(
                        story=story_index,
                        model=model,
                        item=revision_id
                    )

                    print(json.dumps(
                        graph,
                        indent=2,
                        ensure_ascii=False
                    ))

                    model_results[revision_id] = graph

                story_result["models"][model] = model_results

            all_results.append(story_result)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_file = "data/extractions.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()