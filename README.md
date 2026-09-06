# Stability of LLM-Extracted Causal Relations Under Semantic-Preserving Perturbation

This repository contains the code and data for a term paper investigating the stability of causal relations extracted by large language models (LLMs) when narratives are modified by semantic-preserving perturbations.

The project uses MAVEN-ERE stories and their gold causal graphs. For each story, the workflow is:

1. Generate semantic-preserving revisions of the narrative.
2. Extract causal relations from the original and revised narratives with an LLM.
3. Match extracted relations against the gold graph.
4. Compare precision, recall, F1, relation recovery, and agreement across the original and revised narratives.

## Project Structure

```text
causal-stability/
├── data/
│   ├── mavenere_subset_300_context.json  # Stories and gold causal graphs
│   ├── llama_extractions.json             # Original Llama extractions
│   ├── gemma_extractions.json             # Original Gemma extractions
│   ├── qwen_extractions.json              # Original Qwen extractions
│   ├── llama_extractions_revisions.json   # Llama extractions for revisions
│   ├── gemma_extractions_revisions.json   # Gemma extractions for revisions
│   └── qwen_extractions_revisions.json    # Qwen extractions for revisions
├── literature/                            # Notes and data-processing documentation
├── models/                                # Local model files, when used
├── results/                               # Model result summaries
├── src/
│   ├── generate_revisions.py
│   ├── run.py
│   ├── causal_relation_matcher.py
│   ├── causal_relation_matcher_rev.py
│   └── stability_eval.py
├── requirements.txt
└── README.md
```

The repository also contains intermediate datasets, matching outputs, annotation files, and analysis scripts under `data/`, `results/`, and `src/`.

## Installation

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The extraction and revision-generation scripts use Ollama. Install Ollama separately, start the Ollama service, and make sure the required model is available locally. The model names are configured near the top of `src/run.py` and `src/generate_revisions.py`.

For example:

```powershell
ollama pull qwen3:8b
ollama pull llama3.1:8b
```

The embedding-based matchers may use `sentence-transformers` and SciPy. If the embedding model cannot be loaded, the matcher falls back to a lightweight string-similarity implementation.

## Workflow

Run the commands below from the project root.

### 1. Generate semantic-preserving revisions

`generate_revisions.py` creates three revisions for each story:

- `paraphrase`: changes wording while preserving the narrative meaning.
- `reorder`: changes the presentation order of events.
- `context_distance`: inserts neutral contextual sentences between events.

The input path, Ollama model, temperature, and number of stories are configured in the script.

```powershell
python src/generate_revisions.py
```

Generated revisions are stored inside each story's `revisions` array in the configured input JSON file. The script saves after each story and skips stories that already contain revisions.

### 2. Extract causal relations with an LLM

`run.py` sends the original or revised narrative to Ollama and stores the extracted causal relations. The model list and output path are configured near the top of the script.

For revision extraction, configure for example:

```python
MODELS = ["llama3.1:8b"]
OUTPUT_FILE = "data/llama_extractions_revisions.json"
```

Then run:

```powershell
python src/run.py
```

A revision extraction file has the following general structure:

```json
{
  "story_id": "...",
  "original": {
    "text": "...",
    "gold_graph": {"relations": []}
  },
  "revisions": [
    {
      "revision_id": "..._paraphrase",
      "type": "paraphrase",
      "text": "..."
    }
  ],
  "models": {
    "llama3.1:8b": {
      "..._paraphrase": {"relations": []},
      "..._reorder": {"relations": []},
      "..._context_distance": {"relations": []}
    }
  }
}
```

### 3. Match original extractions against the gold graph

`causal_relation_matcher.py` matches the model's extracted relations from the original narrative against the gold causal graph. It reports lexical matches first and uses semantic similarity as a fallback.

```powershell
python src/causal_relation_matcher.py `
    data/mavenere_subset_300_context.json `
    data/llama_extractions.json `
    --threshold 0.5 `
    --out data/matches_llama.json
```

Arguments:

- `gold.json`: gold stories and causal graphs.
- `predictions.json`: model extraction results.
- `--threshold`: semantic matching threshold.
- `--quiet`: suppress pair-level diagnostics.
- `--out`: write aggregated and per-story statistics to a JSON file.

### 4. Match revision extractions against the gold graph

`causal_relation_matcher_rev.py` uses the same matching logic for revision extraction files. It evaluates the revision prediction entries against the same original gold graph.

```powershell
python src/causal_relation_matcher_rev.py `
    data/mavenere_subset_300_context.json `
    data/llama_extractions_revisions.json `
    --threshold 0.5 `
    --out data/matches_llama_revisions.json
```

For separate files per revision type, run the matcher with prediction files filtered to one revision, or use `stability_eval.py`, which evaluates all supported conditions in one run.

### 5. Compute stability across perturbations

`stability_eval.py` compares original extraction results with revision extraction results for each model and story. It evaluates these conditions:

- `original`
- `paraphrase`
- `reorder`
- `context_distance`

It reports metric drift, per-relation recovery, original-versus-revision agreement, flip rates, and lexical-match degradation.

```powershell
python src/stability_eval.py `
    --originals data/gemma_extractions.json data/llama_extractions.json data/qwen_extractions.json `
    --revisions data/gemma_extractions_revisions.json data/llama_extractions_revisions.json data/qwen_extractions_revisions.json `
    --threshold 0.5 `
    --out stability_report.json
```

The model name is inferred from each file's `models` object, so the input files can be supplied in any order.

## Matching Method

For each predicted relation, the matcher compares the predicted cause and effect with the corresponding gold triggers and contexts:

1. Exact trigger matching is attempted first.
2. Semantic similarity is used when exact matching fails.
3. Candidate relations are assigned one-to-one so that a prediction and a gold relation cannot be matched multiple times.

The reported statistics include:

- true positives, false positives, and false negatives;
- precision, recall, and F1;
- lexical matches and semantic fallback matches;
- relation recovery across conditions;
- agreement and flip rates between original and revised narratives.

## Notes

- Run all commands from the repository root so the configured relative paths resolve correctly.
- Ollama must be running before executing the generation or extraction scripts.
- The generation and extraction scripts save intermediate results so interrupted runs can be resumed.
- Gold causal graphs are used for evaluation and are not changed by the matching scripts.
- Before a full experiment, inspect the configured model names and input/output paths in the script constants.
