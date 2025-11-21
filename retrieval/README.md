## Environment setup

- **Python**: 3.10+ recommended.
- **Create a virtual environment and install dependencies** (from the repo root or inside `retrieval/`):

```bash
cd /gpfs/radev/home/jw3278/project/BRIGHT
cd retrieval
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# HuggingFace datasets (for loading BRIGHT-PRO-WITH-ASPECT)
pip install datasets
```

- **Optional GPU support**: install a CUDA-enabled `torch` build appropriate for your system if you want to run heavy encoder models on GPU.
- **External API keys** (only needed for corresponding models):
  - **Azure OpenAI models (`azure_openai`)**:

    ```bash
    export AZURE_OPENAI_ENDPOINT="https://your-azure-endpoint"
    export AZURE_OPENAI_API_KEY="your-azure-openai-key"
    ```

  - Other API-backed models (e.g., Cohere, Vertex AI, etc.) rely on their standard Python client environment variables.

## Running retrieval baselines

The main entry point for running retrieval over BRIGHT-PRO-WITH-ASPECT is `run.py`. It loads the datasets via HuggingFace `datasets`, computes retrieval scores, and writes them to `outputs/{task}_{model}/score.json`.

- **Basic usage** (from `retrieval/`):

```bash
cd /gpfs/radev/home/jw3278/project/BRIGHT/retrieval
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
azure_openai, qwen, qwen2, sbert, sf,
reasonir, diver-retriever, qwen3-embed
```

- **Common optional flags**:
  - **`--long_context`**: use long-context instructions and gold labels (`gold_ids_long`).
  - **`--output_dir`**: base directory for outputs (default: `outputs`).
  - **`--cache_dir`**: cache directory for datasets and embeddings (default: `cache`).
  - **`--config_dir`**: directory containing per-model configuration JSONs (default: `configs`).
  - **`--encode_batch_size`**, **`--query_max_length`**, **`--doc_max_length`**: control encoding efficiency and truncation.
  - **`--key`**: API key override for some models (e.g., Azure OpenAI) instead of using environment variables.
  - **`--model_cache_folder`**: local cache directory for model weights.
  - **`--debug`**: run on a small subset of documents for quick sanity checks.

## Evaluating retrieval runs

Retrieval scores (`score.json` files) can be evaluated with the scripts under `evaluation/`, e.g. `evaluation/alpha-ndcg-evaluation.py` for aspect-aware alpha-nDCG. See that script for detailed CLI options; the typical workflow is:

1. Run `python run.py ...` to generate `outputs/{task}_{model}/score.json`.
2. Call the appropriate script in `evaluation/` pointing to the output directory and task.


