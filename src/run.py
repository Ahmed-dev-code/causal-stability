import json
import os
import ollama
from tqdm import tqdm


MODELS = [
    "gemma3:4b",
]

OUTPUT_FILE = "data/gemma_extractions_revisions.json"


def load_existing_results(path):
    """Load previously saved extraction results so the script can resume."""
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        print(f"Warning: could not read existing results from {path}; starting fresh.")

    return []


def save_results(results, path):
    """Persist current results to disk after each story to allow crash recovery."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


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
- extract the exact text don't parapharase.
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

    try:
        return json.loads(content)

    except json.JSONDecodeError as e:
        print("\n" + "=" * 80)
        print("JSON PARSING FAILED")
        print("=" * 80)
        print("Error:", e)
        print("Raw model response:")
        print(repr(content))
        print("=" * 80)

        return {
            "relations": [],
            "error": "invalid_json",
            "raw_response": content
        }


def main():

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    with open(
        "data/mavenere_subset_300_context.json",
        "r",
        encoding="utf-8"
    ) as f:

        stories = json.load(f)

    existing_results = load_existing_results(OUTPUT_FILE)

    last_completed_story_id = None
    for item in reversed(existing_results):
        if isinstance(item, dict) and item.get("story_id"):
            last_completed_story_id = item["story_id"]
            break

    start_index = 0
    if last_completed_story_id is not None:
        for idx, story in enumerate(stories):
            if story.get("story_id") == last_completed_story_id:
                start_index = idx + 1
                break

    remaining_stories = stories[start_index:]

    print("=" * 80)
    print(f"Loaded {len(stories)} stories")
    print(f"Last saved story: {last_completed_story_id or 'none'}")
    print(f"Resume from index: {start_index}")
    print(f"Remaining to process: {len(remaining_stories)}")
    print(f"Models: {len(MODELS)}")
    print("=" * 80)

    all_results = existing_results
    total_requests = sum(
        len(MODELS) * (1 + len(story["revisions"]))
        for story in remaining_stories
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
        for story_index, story in enumerate(remaining_stories, start=start_index + 1):

            story_id = story["story_id"]

            print("\n")
            print("#" * 80)
            print(
                f"STORY {story_index}/{len(stories)}: "
                f"{story_id}"
            )
            print("#" * 80)

            story_result = {
                "story_id": story_id,
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

                try:
                    print("\n--- ORIGINAL ---")
                    # original_graph = extract_causality(
                    #     model,
                    #     story["text"]
                    # )
                    progress.update(1)
                    progress.set_postfix(
                        story=story_index,
                        model=model,
                        item="original"
                    )

                    # print(json.dumps(
                    #     original_graph,
                    #     indent=2,
                    #     ensure_ascii=False
                    # ))
                    # model_results["original"] = original_graph
                    print("skipped the original text extraction")
                except Exception as exc:
                    print(f"ERROR while processing original for {story_id}: {exc}")
                    model_results["original"] = {
                        "relations": [],
                        "error": str(exc)
                    }

                # --------------------------------------------------
                # Revisions
                # --------------------------------------------------

                for revision in story["revisions"]:

                    revision_id = revision["revision_id"]

                    try:
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
                    except Exception as exc:
                        print(f"ERROR while processing {revision_id} for {story_id}: {exc}")
                        model_results[revision_id] = {
                            "relations": [],
                            "error": str(exc)
                        }

                story_result["models"][model] = model_results

            all_results.append(story_result)
            save_results(all_results, OUTPUT_FILE)

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()