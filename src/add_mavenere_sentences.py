#!/usr/bin/env python3

"""
Add the original MAVEN sentence segmentation to a subset dataset.

Source:
    mavenere/train.jsonl

Target:
    mavenere_subset_300.json

For every story in the subset, find the corresponding MAVEN-ERE
document using:

    subset["story_id"] == train.jsonl["id"]

Then copy:

    train.jsonl["sentences"]

into:

    subset["sentences"]

The original fields in mavenere_subset_300.json are preserved.

Usage:

    python add_maven_sentences.py

Or:

    python add_mavenere_sentences.py \
        mavenere/train.jsonl \
        mavenere_subset_300.json \
        mavenere_subset_300_with_sentences.json
"""

import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Load JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """
    Load a JSON Lines file.

    Returns:
        dict mapping document ID -> document
    """

    documents = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                document = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"WARNING: Could not parse line "
                    f"{line_number}: {e}",
                    file=sys.stderr
                )

                continue

            document_id = document.get("id")

            if document_id is None:

                print(
                    f"WARNING: Line {line_number} "
                    f"has no 'id' field.",
                    file=sys.stderr
                )

                continue

            documents[str(document_id)] = document

    return documents


# ---------------------------------------------------------------------------
# Load subset
# ---------------------------------------------------------------------------

def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def add_sentences(
    train_documents,
    subset
):
    """
    Add MAVEN sentence arrays to the subset.

    Matching:

        story_id -> train.jsonl id
    """

    if isinstance(subset, dict):
        stories = [subset]
    else:
        stories = subset

    matched = 0
    missing = 0
    empty_sentences = 0

    for story in stories:

        story_id = str(
            story.get("story_id", "")
        )

        if not story_id:

            print(
                "WARNING: Story has no story_id.",
                file=sys.stderr
            )

            missing += 1
            continue

        maven_document = train_documents.get(
            story_id
        )

        if maven_document is None:

            print(
                f"WARNING: No MAVEN document found "
                f"for story_id={story_id}",
                file=sys.stderr
            )

            missing += 1
            continue

        sentences = maven_document.get(
            "sentences"
        )

        if not isinstance(
            sentences,
            list
        ):

            print(
                f"WARNING: MAVEN document "
                f"{story_id} has no valid sentences array.",
                file=sys.stderr
            )

            missing += 1
            continue

        if len(sentences) == 0:

            empty_sentences += 1

        # ---------------------------------------------------------------
        # Add the ORIGINAL MAVEN sentence segmentation.
        # ---------------------------------------------------------------

        story["sentences"] = sentences

        matched += 1

    return (
        stories,
        matched,
        missing,
        empty_sentences
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if len(sys.argv) == 1:

        train_path = "data/mavenere/train.jsonl"
        subset_path = "data/mavenere_subset_300.json"
        output_path = (
            "data/mavenere_subset_300_with_sentences.json"
        )

    elif len(sys.argv) == 4:

        train_path = sys.argv[1]
        subset_path = sys.argv[2]
        output_path = sys.argv[3]

    else:

        print(
            "Usage:\n"
            "  python add_maven_sentences.py\n\n"
            "or:\n"
            "  python add_maven_sentences.py "
            "<train.jsonl> <subset.json> <output.json>",
            file=sys.stderr
        )

        sys.exit(1)

    # -----------------------------------------------------------------------
    # Check files.
    # -----------------------------------------------------------------------

    for path in (
        train_path,
        subset_path
    ):

        if not Path(path).exists():

            print(
                f"ERROR: File not found: {path}",
                file=sys.stderr
            )

            sys.exit(1)

    # -----------------------------------------------------------------------
    # Load data.
    # -----------------------------------------------------------------------

    print(
        f"Loading MAVEN-ERE: {train_path}"
    )

    train_documents = load_jsonl(
        train_path
    )

    print(
        f"Loaded {len(train_documents)} "
        f"MAVEN documents."
    )

    print(
        f"Loading subset: {subset_path}"
    )

    subset = load_json(
        subset_path
    )

    # -----------------------------------------------------------------------
    # Add sentences.
    # -----------------------------------------------------------------------

    (
        stories,
        matched,
        missing,
        empty_sentences
    ) = add_sentences(
        train_documents,
        subset
    )

    # -----------------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stories,
            f,
            ensure_ascii=False,
            indent=2
        )

    # -----------------------------------------------------------------------
    # Report.
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Subset stories:       {len(stories)}"
    )

    print(
        f"MAVEN matched:        {matched}"
    )

    print(
        f"MAVEN missing:        {missing}"
    )

    print(
        f"Empty sentence lists: {empty_sentences}"
    )

    print(
        f"Output:               {output_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
