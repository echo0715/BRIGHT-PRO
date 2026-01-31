### Environment setup

- **Python version**: Python 3.10.
- **Install with `uv` (recommended)**:

```bash
pip install uv  # if not already installed
uv sync
```

- **Install with plain `pip`** (alternative):

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
```

- **Required environment variables** (Azure OpenAI):

```bash
export AZURE_KEY="your-azure-openai-key"
export AZURE_ENDPOINT="https://your-azure-endpoint"
```

### Running retrieval agents

Retrieval agents live under `search_agent/` and write run files to `runs/gpt-5-mini/...`. BRIGHT-PRO data is loaded via `datasets` and cached under `cache/` by default.

- **Run adaptive round the Azure OpenAI search agent** :

```bash
python -m search_agent.openai_new \
  --task biology \
  --searcher-type bm25 \
  --model gpt-5-mini \
```

- **Run the fixed-turn retrieval OpenAI agent**:
```bash
python -m search_agent.openai_fixed_turn \
  --task biology \
  --searcher-type bm25 \
  --model gpt-5-mini \
  --output-dir fixed_turn_runs/gpt-5-mini \
  --cache-dir cache
```



Replace `--task` with one of:

```text
biology, earth_science, economics, psychology, robotics,
stackoverflow, sustainable_living
```

Replace `--searcher-type` with one of:

```text
bm25, reasonir, diver, bge, inst-l, inst-xl, qwen, grit
```

Models and document embeddings are cached under `cache/` and `cache/doc_emb/...` as needed; you can override with `--model-cache-dir` or `--cache-dir`.

### Generating answers from prior runs
Generate full answers based on stored fixed turn retrieval runs using Azure OpenAI with `search_agent/answers_from_runs.py`:

```bash
python -m search_agent.answers_from_runs \
  --searcher bm25 \
  --task biology \
  --model gpt-5-mini \
```


### Evaluating retrieval (alpha-nDCG)

Alpha-nDCG is the main metric used in the paper; it evaluates aspect-aware, diversified retrieval over **fixed-turn** runs.

- **Compute alpha-nDCG for fixed-turn runs**:

```bash
python -m scripts_evaluation.alpha_ndcg \
  --fixed_dir fixed_turn_runs \
  --alpha 0.5 \
```

Key flags:
- **`--benchmark`**: optional filter for the generator/benchmark subfolder (e.g., `--benchmark bge`).
- **`--save_json`** / **`--excel_out`**: optionally save detailed results and model×task matrices.

