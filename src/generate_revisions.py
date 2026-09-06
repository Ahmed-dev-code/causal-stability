#!/usr/bin/env python3

"""
Generate semantic-preserving revisions for MAVEN-ERE stories.

Input:
    data/mavenere.json (or any JSON file with stories)

Output:
    Revisions are saved back to the same input file in a "revisions" array.
    File structure:
    {
        "stories": [...],
        "revisions": [
            {
                "story_id": "...",
                "title": "...",
                "original_text": "...",
                "revisions": {...}
            }
        ]
    }

Revision types:
    1. paraphrase
    2. reorder
    3. context_distance

The original gold graph is NOT modified.

The script is resumable:
    - already generated stories are skipped
    - results are saved after every story

Requirements:
    pip install ollama

Ollama must be running locally on default port (11434).

Example:
    python src/generate_revisions.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ollama


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "mavenere_subset_300_context.json"

# Use the model you already have locally.
MODEL = "qwen3:8b"

# Keep generation deterministic.
TEMPERATURE = 0.1

# Number of additional neutral sentences for
# context-distance revision.
NUM_CONTEXT_SENTENCES = 2

# Set to None to process all stories.
MAX_STORIES = None


# ============================================================
# OLLAMA
# ============================================================

def call_ollama(
    prompt: str,
    model: str = MODEL,
) -> str:

    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        format="json",
        options={
            "temperature": TEMPERATURE,
        },
    )

    return response["response"]


# ============================================================
# JSON PARSING
# ============================================================

def parse_json_response(
    response: str,
) -> Dict[str, Any]:

    try:
        result = json.loads(response)

        if not isinstance(result, dict):
            raise ValueError(
                "LLM response is not a JSON object."
            )

        return result

    except json.JSONDecodeError as exc:

        print("\nInvalid JSON returned by model:")
        print(response)

        raise ValueError(
            f"Could not parse LLM JSON response: {exc}"
        )


# ============================================================
# PARAPHRASE
# ============================================================

def generate_paraphrase(
    title: str,
    text: str,
) -> str:

    prompt = f"""
You are generating a semantic-preserving revision for a causal
relation extraction robustness experiment.

TITLE:
{title}

ORIGINAL NARRATIVE:
{text}

TASK:

Rewrite the entire narrative as a paraphrase.

STRICT REQUIREMENTS:

1. Preserve EVERY factual statement.
2. Preserve EVERY event.
3. Preserve EVERY entity, person, organization, place and object.
4. Preserve dates, numbers and quantities.
5. Preserve temporal information.
6. Preserve all causal relationships.
7. Preserve the direction of every causal relationship.
8. Do NOT add new events.
9. Do NOT remove events.
10. Do NOT merge distinct events.
11. Do NOT split one event into multiple events.
12. Do NOT introduce new causal relationships.
13. Do NOT change factual meaning.
14. Change wording and sentence structure substantially.
15. The result should remain natural English.
16. Do not mention this instruction or the experiment.

Return ONLY this JSON:

{{
  "text": "the complete paraphrased narrative"
}}
"""

    response = call_ollama(prompt)

    result = parse_json_response(response)

    revised = result.get("text")

    if not revised:
        raise ValueError(
            "Paraphrase response did not contain 'text'."
        )

    return revised.strip()


# ============================================================
# REORDER
# ============================================================

def generate_reorder(
    title: str,
    text: str,
) -> str:

    prompt = f"""
You are generating a semantic-preserving revision for a causal
relation extraction robustness experiment.

TITLE:
{title}

ORIGINAL NARRATIVE:
{text}

TASK:

Rewrite the narrative while changing the PRESENTATION ORDER of
events.

The purpose is to test whether a causal extraction model depends
on the order in which events are presented.

STRICT REQUIREMENTS:

1. Preserve EVERY factual statement.
2. Preserve EVERY event.
3. Preserve EVERY entity, person, organization, place and object.
4. Preserve dates, numbers and quantities.
5. Preserve all causal relationships.
6. Preserve the direction of every causal relationship.
7. Do NOT add events.
8. Do NOT remove events.
9. Do NOT change factual meaning.
10. Do NOT invent information.
11. Where naturally possible, present an effect before its cause.
12. Change the order of relevant events rather than merely
    changing a few words.
13. Keep the resulting narrative grammatically coherent.
14. Preserve temporal facts. If necessary, use expressions such
    as "earlier", "previously", "later", or "the following day"
    so that chronology remains factually correct.
15. Do not introduce new causal relationships.
16. Do not mention this instruction or the experiment.

IMPORTANT:

The goal is NOT random sentence shuffling.

The resulting text must be a coherent narrative in which
causal information is presented in a different order.

Return ONLY this JSON:

{{
  "text": "the complete reordered narrative"
}}
"""

    response = call_ollama(prompt)

    result = parse_json_response(response)

    revised = result.get("text")

    if not revised:
        raise ValueError(
            "Reorder response did not contain 'text'."
        )

    return revised.strip()


# ============================================================
# CONTEXT / DISTANCE
# ============================================================

def generate_context_distance(
    title: str,
    text: str,
) -> str:

    prompt = f"""
You are generating a semantic-preserving revision for a causal
relation extraction robustness experiment.

TITLE:
{title}

ORIGINAL NARRATIVE:
{text}

TASK:

Create a revised version of the narrative that increases the
amount of irrelevant contextual information between important
events.

The purpose is to test whether a causal extraction model becomes
less stable when causally related events are separated by
distracting information.

STRICT REQUIREMENTS:

1. Preserve EVERY original factual statement.
2. Preserve EVERY original event.
3. Preserve EVERY original entity.
4. Preserve dates, numbers and quantities.
5. Preserve EVERY original causal relationship.
6. Preserve causal direction.
7. Do NOT remove original information.
8. Do NOT change factual meaning.
9. Do NOT alter the causal relationships in the original story.
10. Add approximately {NUM_CONTEXT_SENTENCES} short,
    neutral, contextually plausible sentences.
11. Added sentences must NOT introduce causal relationships
    involving the original causal events.
12. Added sentences must NOT contradict the story.
13. Added sentences must NOT introduce important new events.
14. Added sentences should function as distractors.
15. Place the added sentences between existing events where
    possible, especially between events that participate in
    causal relations.
16. Keep the narrative natural and coherent.
17. Do not mention this instruction or the experiment.

IMPORTANT:

The inserted sentences should make the relevant events
textually farther apart without changing their meaning.

Return ONLY this JSON:

{{
  "text": "the complete revised narrative"
}}
"""

    response = call_ollama(prompt)

    result = parse_json_response(response)

    revised = result.get("text")

    if not revised:
        raise ValueError(
            "Context-distance response did not contain 'text'."
        )

    return revised.strip()


# ============================================================
# LOAD INPUT
# ============================================================

def load_input_file(
    path: Path,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Load input file and return (stories, full_data_dict)."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Support both:
    #
    # [
    #   {...},
    #   {...}
    # ]
    #
    # and:
    #
    # {
    #   "stories": [...],
    #   "revisions": []
    # }
    # --------------------------------------------------------

    if isinstance(data, list):
        stories = data
        full_data = {
            "stories": stories,
            "revisions": [],
        }

    elif isinstance(data, dict):
        if isinstance(
            data.get("stories"),
            list,
        ):
            stories = data["stories"]
            full_data = data

        else:
            # If the file itself contains one story.
            if "story_id" in data:
                stories = [data]
                full_data = {
                    "stories": stories,
                    "revisions": [],
                }
            else:
                raise ValueError(
                    "Could not find stories in input JSON."
                )

    else:
        raise ValueError(
            "Unexpected input JSON structure."
        )

    # Ensure revisions array exists
    if "revisions" not in full_data:
        full_data["revisions"] = []

    return stories, full_data



def load_existing_revisions(
    full_data: Dict[str, Any],
) -> set[str]:
    """Return the set of story_ids that already have revisions nested in each story."""

    stories = full_data.get("stories", [])

    if not isinstance(stories, list):
        return set()

    existing = set()

    for story in stories:
        if not isinstance(story, dict):
            continue

        story_id = story.get("story_id")
        revisions = story.get("revisions", [])

        if story_id is not None and isinstance(revisions, list) and revisions:
            existing.add(str(story_id))

    return existing



def save_to_input(
    path: Path,
    data: Dict[str, Any],
) -> None:
    """Save the updated data back to the input file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    temporary.replace(path)



# ============================================================
# VALIDATION
# ============================================================

def validate_revision(
    original: str,
    revision: str,
) -> None:

    if not revision.strip():

        raise ValueError(
            "Generated revision is empty."
        )

    # Very basic sanity checks.
    #
    # We intentionally do not automatically decide whether the
    # revision is semantically correct. That requires semantic
    # validation and/or manual inspection.
    if len(revision.split()) < 20:

        raise ValueError(
            "Generated revision is suspiciously short."
        )

    # Detect obvious refusal responses.
    refusal_markers = [
        "I cannot",
        "I can't",
        "I am unable",
        "cannot comply",
    ]

    lower = revision.lower()

    for marker in refusal_markers:

        if marker.lower() in lower:

            raise ValueError(
                "Model appears to have returned a refusal."
            )


# ============================================================
# GENERATE ONE STORY
# ============================================================

def generate_story_revisions(
    story: Dict[str, Any],
) -> list[Dict[str, Any]]:

    story_id = story.get(
        "story_id"
    )

    title = story.get(
        "title",
        "",
    )

    text = story.get(
        "text",
        "",
    )

    if not story_id:

        raise ValueError(
            "Story has no story_id."
        )

    if not text:

        raise ValueError(
            f"Story {story_id} has no text."
        )

    print()
    print("=" * 80)
    print(
        f"Story: {story_id}"
    )
    print(
        f"Title: {title}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Generate paraphrase
    # --------------------------------------------------------

    print(
        "\n[1/3] Generating paraphrase..."
    )

    paraphrase = generate_paraphrase(
        title,
        text,
    )

    validate_revision(
        text,
        paraphrase,
    )

    print(
        "      ✓ paraphrase generated"
    )

    # --------------------------------------------------------
    # Generate reorder
    # --------------------------------------------------------

    print(
        "\n[2/3] Generating event-order revision..."
    )

    reorder = generate_reorder(
        title,
        text,
    )

    validate_revision(
        text,
        reorder,
    )

    print(
        "      ✓ reorder generated"
    )

    # --------------------------------------------------------
    # Generate context-distance
    # --------------------------------------------------------

    print(
        "\n[3/3] Generating context-distance revision..."
    )

    context_distance = (
        generate_context_distance(
            title,
            text,
        )
    )

    validate_revision(
        text,
        context_distance,
    )

    print(
        "      ✓ context-distance generated"
    )

    # --------------------------------------------------------
    # Save the generated revisions to the story's revisions list.
    # --------------------------------------------------------

    return [
        {
            "revision_id": f"{story_id}_paraphrase",
            "type": "paraphrase",
            "text": paraphrase,
        },
        {
            "revision_id": f"{story_id}_reorder",
            "type": "reorder",
            "text": reorder,
        },
        {
            "revision_id": f"{story_id}_context_distance",
            "type": "context_distance",
            "text": context_distance,
        },
    ]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("MAVEN-ERE REVISION GENERATOR")
    print("=" * 80)

    print(
        f"\nInput:  {INPUT_PATH}"
    )

    print(
        f"Model:  {MODEL}"
    )

    # --------------------------------------------------------
    # Load input file
    # --------------------------------------------------------

    stories, full_data = load_input_file(
        INPUT_PATH
    )

    print(
        f"\nLoaded {len(stories)} stories."
    )

    if MAX_STORIES is not None:

        stories = stories[
            :MAX_STORIES
        ]

        print(
            f"Limited to {len(stories)} stories."
        )

    # --------------------------------------------------------
    # Check for existing revisions
    # --------------------------------------------------------

    existing = load_existing_revisions(
        full_data
    )

    print(
        f"Already generated: "
        f"{len(existing)} stories."
    )

    # --------------------------------------------------------
    # Process stories
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0

    for index, story in enumerate(
        stories,
        start=1,
    ):

        story_id = str(
            story.get("story_id")
        )

        print()
        print(
            f"[{index}/{len(stories)}] "
            f"{story_id}"
        )

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if story_id in existing:

            print(
                "Already generated. Skipping."
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        try:

            generated_revisions = (
                generate_story_revisions(
                    story
                )
            )

            story.setdefault(
                "revisions",
                [],
            )
            story["revisions"].extend(
                generated_revisions
            )

            # Save immediately.
            save_to_input(
                INPUT_PATH,
                full_data,
            )

            processed += 1

            print(
                "\n✓ Saved successfully."
            )

        except Exception as exc:

            failed += 1

            print(
                "\n✗ ERROR:"
            )

            print(
                str(exc)
            )

            print(
                "Story skipped; continuing."
            )

            # Small pause to avoid hammering Ollama
            # after an error.
            time.sleep(1)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINISHED")
    print("=" * 80)

    print(
        f"Generated: {processed}"
    )

    print(
        f"Skipped:   {skipped}"
    )

    print(
        f"Failed:    {failed}"
    )

    total_revisions = sum(
        len(story.get("revisions", []))
        for story in full_data.get("stories", [])
        if isinstance(story, dict)
    )

    print(
        f"Total revisions: {total_revisions}"
    )

    print(
        f"\nSaved to:\n{INPUT_PATH}"
    )


if __name__ == "__main__":
    main()
