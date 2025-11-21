## Environment setup

- **Python**: 3.10+ recommended.
- **Create a virtual environment and install dependencies** (from the repo root or inside `retrieval/`):

```bash
cd retrieval
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **External API keys** (only needed for corresponding models):
  - **Azure OpenAI models (`azure_openai`)**:

    ```bash
    export AZURE_OPENAI_ENDPOINT="https://your-azure-endpoint"
    export AZURE_OPENAI_API_KEY="your-azure-openai-key"
    ```

## Running retrieval baselines

The main entry point for running retrieval over BRIGHT-PRO is `run.py`. It loads the datasets via HuggingFace `datasets`, computes retrieval scores, and writes them to `outputs/{task}_{model}/score.json`.

- **Basic usage** (from `retrieval/`):

```bash
cd retrieval
python run.py \
  --task biology \
  --model bm25
```

- **Supported tasks** (argument `--task`):

```text
biology, earth_science, economics, psychology, robotics,
stackoverflow, sustainable_living
```

- **Supported models** (argument `--model`):

```text
bm25, grit, inst-l, inst-xl,
azure_openai, qwen2, reasonir, diver-retriever, qwen3-embed
```

- **Common optional flags**:
  - **`--output_dir`**: base directory for outputs (default: `outputs`).
  - **`--cache_dir`**: cache directory for datasets and embeddings (default: `cache`).
  - **`--config_dir`**: directory containing per-model configuration JSONs (default: `configs`).
  - **`--encode_batch_size`**, **`--query_max_length`**, **`--doc_max_length`**: control encoding efficiency and truncation.
  - **`--key`**: API key override for some models (e.g., Azure OpenAI) instead of using environment variables.
  - **`--model_cache_folder`**: local cache directory for model weights.

## Evaluating retrieval runs

Retrieval scores (`score.json` files) can be evaluated with the scripts under `evaluation/`, e.g. `evaluation/alpha-ndcg-evaluation.py` for aspect-aware alpha-nDCG. See that script for detailed CLI options; the typical workflow is:

1. Run `python run.py ...` to generate `outputs/{task}_{model}/score.json`.
2. Call the appropriate script in `evaluation/` pointing to the output directory and task.


