import json
from pathlib import Path

from evaluate import evaluate_graph_pair


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():

    with open(DATA_DIR / "test_extractions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for model, graphs in data["models"].items():

        original = graphs["original"]

        for revision_id in [
            "paraphrase",
            "causal_change",
            "direction_change"
        ]:

            revision = graphs[revision_id]

            evaluation = evaluate_graph_pair(
                original,
                revision
            )

            results.append({
                "model": model,
                "revision": revision_id,
                "event_preservation":
                    evaluation["event_preservation"],
                "edge_preservation":
                    evaluation["edge_preservation"],
                "causal_stability":
                    evaluation["causal_stability"]
            })

    print("\n")
    print("=" * 80)
    print("CAUSAL STABILITY RESULTS")
    print("=" * 80)

    print(
        f"{'Model':<18}"
        f"{'Revision':<20}"
        f"{'Events':<10}"
        f"{'Edges':<10}"
        f"{'Stability':<10}"
    )

    print("-" * 80)

    for result in results:

        print(
            f"{result['model']:<18}"
            f"{result['revision']:<20}"
            f"{result['event_preservation']:<10}"
            f"{result['edge_preservation']:<10}"
            f"{result['causal_stability']:<10}"
        )

    with open(
        DATA_DIR / "stability_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print("\nSaved:")
    print(DATA_DIR / "stability_results.json")


if __name__ == "__main__":
    main()