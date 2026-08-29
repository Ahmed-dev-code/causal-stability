#!/usr/bin/env python3
"""
Enrich gold_graph relations of a MAVEN-ERE-derived JSON dataset.

For every gold causal relation, add:

    type:
        {
            "source_event_type": ...,
            "target_event_type": ...
        }

    context:
        {
            "cause": "...context containing the cause trigger...",
            "effect": "...context containing the effect trigger..."
        }

The cause and effect contexts are generated independently.

The cause context is based on source_sent_id and source_trigger.
The effect context is based on target_sent_id and target_trigger.

Two context modes are supported:

    sentence
        Use the complete sentence containing the trigger.

    window
        Use a token window centered around the trigger. The trigger is
        guaranteed to remain inside the returned window whenever possible.

Example:

    python enrich_gold_graph.py \
        mavenere_subset_300.json \
        mavenere_subset_300_enriched.json

Window mode:

    python enrich_gold_graph.py \
        mavenere_subset_300.json \
        mavenere_subset_300_enriched.json \
        window \
        20

The final argument is the number of tokens on each side of the trigger.

For example, window_size=20 produces approximately:

    20 tokens + trigger + 20 tokens

Matching gold_graph.relations[i] to maven_graph.relations[i] is
positional when both lists have the same length. If the lengths differ,
the script falls back to matching by:

    source_trigger
    target_trigger
    relation_type
"""

import json
import re
import sys


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z0-9"])'
)


def split_sentences(story):
    """
    Get the original MAVEN sentence segmentation.

    The sentences array is copied directly from MAVEN-ERE
    (mavenere/train.jsonl), so its indices correspond exactly
    to source_sent_id / target_sent_id in maven_graph.

    Args:
        story: Story dictionary containing a "sentences" field.

    Returns:
        List[str]
    """

    sentences = story.get("sentences")

    if not isinstance(sentences, list):
        raise ValueError(
            f"Story {story.get('story_id', '<unknown>')} "
            "does not contain a valid 'sentences' array."
        )

    return [
        sentence.strip()
        for sentence in sentences
        if isinstance(sentence, str) and sentence.strip()
    ]


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text):
    """
    Simple whitespace/token punctuation tokenizer.

    We keep punctuation attached to words where possible because this
    context is intended for LLM input rather than linguistic parsing.
    """

    return text.split()


# ---------------------------------------------------------------------------
# Trigger matching
# ---------------------------------------------------------------------------

def find_trigger_tokens(sentence, trigger):
    """
    Find the trigger inside a sentence.

    Matching strategy:

    1. Exact case-sensitive substring
    2. Case-insensitive substring

    Returns:
        (start_char, end_char)

    or:

        None
    """

    if not sentence or not trigger:
        return None

    # Exact match
    start = sentence.find(trigger)

    if start != -1:
        return start, start + len(trigger)

    # Case-insensitive match
    match = re.search(
        re.escape(trigger),
        sentence,
        flags=re.IGNORECASE
    )

    if match:
        return match.start(), match.end()

    return None


def find_trigger_token_index(sentence, trigger):
    """
    Find the token index of the trigger.

    The returned index corresponds to the first token overlapping
    the trigger.

    Returns:
        int

    or:

        None
    """

    match = find_trigger_tokens(sentence, trigger)

    if match is None:
        return None

    trigger_start, trigger_end = match

    tokens = list(
        re.finditer(r'\S+', sentence)
    )

    for index, token_match in enumerate(tokens):

        token_start = token_match.start()
        token_end = token_match.end()

        # Check whether the token overlaps the trigger.
        if (
            token_start < trigger_end
            and token_end > trigger_start
        ):
            return index

    return None


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------

def build_sentence_context(sentences, sent_id):
    """
    Return the complete sentence containing the trigger.

    MAVEN sentence IDs are assumed to be zero-based.
    """

    if sent_id is None:
        return None

    try:
        sent_id = int(sent_id)
    except (TypeError, ValueError):
        return None

    if sent_id < 0 or sent_id >= len(sentences):
        return None

    return sentences[sent_id]


def build_window_context(sentences, sent_id, trigger, window_size=20):
    """
    Return a token window centered around the trigger.

    Example with window_size=5:

        ... token token [TRIGGER] token token ...

    The function attempts to keep the trigger inside the window.

    If the sentence is shorter than the requested window, the entire
    sentence is returned.

    If the trigger cannot be found, the function falls back to the
    complete sentence.
    """

    if sent_id is None:
        return None

    try:
        sent_id = int(sent_id)
    except (TypeError, ValueError):
        return None

    if sent_id < 0 or sent_id >= len(sentences):
        return None

    sentence = sentences[sent_id]

    if not sentence:
        return None

    tokens = tokenize(sentence)

    if len(tokens) <= (2 * window_size + 1):
        return sentence

    trigger_index = find_trigger_token_index(
        sentence,
        trigger
    )

    # If the trigger cannot be located, use a centered sentence window
    # rather than failing completely.
    if trigger_index is None:
        start = 0
        end = min(
            len(tokens),
            2 * window_size + 1
        )

        return " ".join(tokens[start:end])

    # Initial window centered on trigger.
    start = max(
        0,
        trigger_index - window_size
    )

    end = min(
        len(tokens),
        trigger_index + window_size + 1
    )

    # ---------------------------------------------------------------
    # Adjust boundaries so that we still have a reasonably sized
    # window when the trigger is close to the beginning/end.
    # ---------------------------------------------------------------

    desired_length = min(
        len(tokens),
        2 * window_size + 1
    )

    current_length = end - start

    if current_length < desired_length:

        # Need more tokens on the right.
        extra_right = min(
            desired_length - current_length,
            len(tokens) - end
        )

        end += extra_right

        current_length = end - start

    if current_length < desired_length:

        # Still need more tokens; take them from the left.
        extra_left = min(
            desired_length - current_length,
            start
        )

        start -= extra_left

    context_tokens = tokens[start:end]

    context = " ".join(context_tokens)

    return context


def build_trigger_context(
    sentences,
    sent_id,
    trigger,
    mode="sentence",
    window_size=20
):
    """
    Build context for one trigger.
    """

    if mode == "sentence":
        return build_sentence_context(
            sentences,
            sent_id
        )

    if mode == "window":
        return build_window_context(
            sentences,
            sent_id,
            trigger,
            window_size
        )

    raise ValueError(
        f"Unknown context mode: {mode}"
    )


# ---------------------------------------------------------------------------
# MAVEN relation matching
# ---------------------------------------------------------------------------

def match_maven_relation(
    gold_rel,
    maven_relations,
    used_positional,
    idx
):
    """
    Prefer positional matching.

    If positional matching is unavailable, fall back to:

        source_trigger == cause
        target_trigger == effect
        relation_type == relation_type
    """

    if used_positional and idx < len(maven_relations):
        return maven_relations[idx]

    for maven_rel in maven_relations:

        if (
            maven_rel.get("source_trigger")
            == gold_rel.get("cause")
            and
            maven_rel.get("target_trigger")
            == gold_rel.get("effect")
            and
            maven_rel.get("relation_type")
            == gold_rel.get("relation_type")
        ):
            return maven_rel

    return None


# ---------------------------------------------------------------------------
# Story enrichment
# ---------------------------------------------------------------------------

def enrich_story(
    story,
    context_mode="sentence",
    window_size=20
):
    """
    Enrich all gold causal relations in one story.
    """

    text = story.get("text", "")

    sentences = split_sentences(story)

    gold_relations = (
        story
        .get("gold_graph", {})
        .get("relations", [])
    )

    maven_relations = (
        story
        .get("maven_graph", {})
        .get("relations", [])
    )

    positional_ok = (
        len(gold_relations)
        == len(maven_relations)
    )

    for idx, gold_rel in enumerate(gold_relations):

        maven_rel = match_maven_relation(
            gold_rel,
            maven_relations,
            positional_ok,
            idx
        )

        # ---------------------------------------------------------------
        # No matching MAVEN relation
        # ---------------------------------------------------------------

        if maven_rel is None:

            gold_rel["type"] = None

            gold_rel["context"] = {
                "cause": None,
                "effect": None
            }

            continue

        # ---------------------------------------------------------------
        # Semantic event types
        # ---------------------------------------------------------------

        gold_rel["type"] = {
            "source_event_type": (
                maven_rel.get("source_event_type")
            ),
            "target_event_type": (
                maven_rel.get("target_event_type")
            )
        }

        # ---------------------------------------------------------------
        # Cause context
        # ---------------------------------------------------------------

        cause_trigger = (
            gold_rel.get("cause")
            or maven_rel.get("source_trigger")
        )

        cause_sent_id = maven_rel.get(
            "source_sent_id"
        )

        cause_context = build_trigger_context(
            sentences,
            cause_sent_id,
            cause_trigger,
            mode=context_mode,
            window_size=window_size
        )

        # ---------------------------------------------------------------
        # Effect context
        # ---------------------------------------------------------------

        effect_trigger = (
            gold_rel.get("effect")
            or maven_rel.get("target_trigger")
        )

        effect_sent_id = maven_rel.get(
            "target_sent_id"
        )

        effect_context = build_trigger_context(
            sentences,
            effect_sent_id,
            effect_trigger,
            mode=context_mode,
            window_size=window_size
        )

        # ---------------------------------------------------------------
        # Store separate cause/effect contexts
        # ---------------------------------------------------------------

        gold_rel["context"] = {
            "cause": cause_context,
            "effect": effect_context
        }

    return story


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    if len(sys.argv) not in (3, 4, 5):

        print(
            "Usage:\n"
            "  python enrich_gold_graph.py "
            "input.json output.json "
            "[sentence|window] [window_size]\n\n"

            "Examples:\n"
            "  python enrich_gold_graph.py "
            "mavenere_subset_300.json "
            "mavenere_subset_300_enriched.json\n\n"

            "  python enrich_gold_graph.py "
            "mavenere_subset_300.json "
            "mavenere_subset_300_enriched.json "
            "sentence\n\n"

            "  python enrich_gold_graph.py "
            "mavenere_subset_300.json "
            "mavenere_subset_300_enriched.json "
            "window 20",

            file=sys.stderr
        )

        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    context_mode = (
        sys.argv[3]
        if len(sys.argv) >= 4
        else "sentence"
    )

    window_size = (
        int(sys.argv[4])
        if len(sys.argv) >= 5
        else 20
    )

    if context_mode not in (
        "sentence",
        "window"
    ):

        print(
            f"Unknown context mode '{context_mode}'. "
            f"Expected 'sentence' or 'window'.",
            file=sys.stderr
        )

        sys.exit(1)

    if window_size <= 0:

        print(
            "window_size must be greater than 0.",
            file=sys.stderr
        )

        sys.exit(1)

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    with open(
        in_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    stories = (
        data
        if isinstance(data, list)
        else [data]
    )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    n_stories = 0
    n_relations = 0
    n_missing_match = 0
    n_missing_cause_context = 0
    n_missing_effect_context = 0

    # -----------------------------------------------------------------------
    # Enrich
    # -----------------------------------------------------------------------

    for story in stories:

        enrich_story(
            story,
            context_mode=context_mode,
            window_size=window_size
        )

        n_stories += 1

        relations = (
            story
            .get("gold_graph", {})
            .get("relations", [])
        )

        for relation in relations:

            n_relations += 1

            context = relation.get(
                "context",
                {}
            )

            if relation.get("type") is None:
                n_missing_match += 1

            if context.get("cause") is None:
                n_missing_cause_context += 1

            if context.get("effect") is None:
                n_missing_effect_context += 1

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    with open(
        out_path,
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
    # Report
    # -----------------------------------------------------------------------

    print(
        f"Done.\n"
        f"  Stories:                 {n_stories}\n"
        f"  Gold relations:          {n_relations}\n"
        f"  Missing MAVEN matches:   {n_missing_match}\n"
        f"  Missing cause context:   {n_missing_cause_context}\n"
        f"  Missing effect context:  {n_missing_effect_context}\n"
        f"  Context mode:            {context_mode}\n"
        f"  Window size:             {window_size}\n"
        f"  Output:                  {out_path}",
        file=sys.stderr
    )


if __name__ == "__main__":
    main()
