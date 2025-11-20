## BRIGHT Retrieval Benchmark

Code for running and evaluating retrieval models on the BRIGHT dataset with aspect-aware evaluation metrics and long-context support.

### Overview

BRIGHT is a multi-domain retrieval benchmark built on the Hugging Face dataset `ya-ir/BRIGHT-PRO-WITH-ASPECT`.  
Each query comes with:

- **Gold documents** (short and long-context variants)
- **Per-document aspects** (e.g., concepts, subtopics)
- **Aspect weights**, enabling evaluation of both **relevance** and **coverage/diversity** of aspects.

This repository provides:

- **Unified retrieval script** (`run.py`) for multiple domains and retriever models
- **Caching** for document/query embeddings
- **Aspect-aware metrics**:
  - `alpha-ndcg-evaluation.py` – weighted alpha-nDCG@K
  - `weighted_aspect_recall.py` – Weighted Aspect Recall@K

If you use this code or the BRIGHT dataset in your work, please consider citing the corresponding paper (fill in your citation below).

### Tasks and Domains

The main script currently supports the following tasks via `--task`:

- **biology**
- **earth_science**
- **economics**
- **psychology**
- **robotics**
- **stackoverflow**
- **sustainable_living**

The evaluation scripts also support additional tasks (e.g., `pony`, `aops`, `leetcode`, `theoremqa_theorems`, `theoremqa_questions`) if you have generated runs for them.

### Supported Retrievers

The `--model` flag in `run.py` selects among several retrievers (dense, sparse, API-based, etc.):

- **bm25**
- **grit**
- **inst-l**
- **inst-xl**
- **azure_openai**
- **qwen**
- **qwen2**
- **sbert**
- **sf**
- **reasonir**
- **diver-retriever**
- **qwen3-embed**

Model-specific configuration (e.g., prompts, instructions, hyperparameters) is stored under `configs/<model>/<task>.json`.

### Installation

- **Clone the repository**

```bash
git clone <your-repo-url>.git
cd BRIGHT
```

- **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate
```

- **Install dependencies**

```bash
pip install -r requirements.txt
```

Some retrievers require additional credentials or environment variables (e.g., OpenAI, Azure OpenAI, Vertex AI, Cohere, VoyageAI).  
Configure those according to your account setup (e.g., `OPENAI_API_KEY`, Azure endpoint and key, etc.).

### Dataset

All scripts load data via the Hugging Face Datasets hub:

- Dataset: `ya-ir/BRIGHT-PRO-WITH-ASPECT`
- Splits:
  - `examples` – queries, gold document ids (short and long variants), and exclusion lists
  - `documents` – document ids, contents, and associated aspect ids
  - `aspects` – aspect ids and their weights

On first run, the dataset will be downloaded and cached under the `cache/` directory (configurable with `--cache_dir`).

### Running Retrieval

The main entrypoint is `run.py`. It:

1. Loads queries and documents for a given `--task`
2. Applies the selected `--model` using the corresponding config in `configs/`
3. Caches embeddings/scores where appropriate
4. Writes per-query retrieval scores to `score.json` under the specified `--output_dir`

Basic example (dense retriever with long-context instructions):

```bash
python run.py \
  --task biology \
  --model qwen2 \
  --long_context \
  --encode_batch_size 8 \
  --output_dir outputs \
  --cache_dir cache \
  --config_dir configs \
  --model_cache_folder /path/to/hf_cache
```

Key arguments:

- **`--task`**: one of `biology`, `earth_science`, `economics`, `psychology`, `robotics`, `stackoverflow`, `sustainable_living`
- **`--model`**: one of the retrievers listed above
- **`--long_context`**: use long-context instructions and long gold labels (`gold_ids_long`)
- **`--query_max_length`, `--doc_max_length`**: optional token length caps for encoders
- **`--encode_batch_size`**: batch size for embedding/encoding
- **`--checkpoint`**: optional path/checkpoint for local models
- **`--key`**: API key for services that are passed via arguments
- **`--model_cache_folder`**: local Hugging Face cache directory for model weights
- **`--input_file`**: use a custom JSON file of examples instead of loading from the dataset

Output files are written to:

- `outputs/<task>_<model>_long_<True|False>/score.json`

The `score.json` file maps each query id to a dict of `{doc_id: score, ...}` or to a ranked list of document ids, depending on the retriever implementation.

### Evaluating Retrieval

This repository includes two evaluation scripts for aspect-aware metrics.

#### Weighted Aspect Recall@K

Script: `evaluation/weighted_aspect_recall.py`

Example (single task):

```bash
python evaluation/weighted_aspect_recall.py \
  --task biology \
  --k 25 \
  --output_dir outputs \
  --cache_dir cache \
  --long_context
```

Example (aggregate across all tasks / runs in `outputs/`):

```bash
python evaluation/weighted_aspect_recall.py \
  --task all \
  --k 25 \
  --output_dir outputs \
  --cache_dir cache \
  --all
```

Useful flags:

- **`--score_file`**: explicit path to a `score.json` file (otherwise inferred from `--task` and `--output_dir`)
- **`--save_json`**: path to write a JSON summary of results
- **`--save_excel`**: path to write an Excel matrix (retrievers × tasks); requires `pandas` and `openpyxl`

The script prints per-run and aggregate Weighted Aspect Recall@K, and can also summarize averages per retriever across tasks.

#### Weighted alpha-nDCG@K

Script: `evaluation/alpha-ndcg-evaluation.py`

Example:

```bash
python evaluation/alpha-ndcg-evaluation.py \
  --task biology \
  --k 25 \
  --alpha 0.5 \
  --output_dir outputs \
  --cache_dir cache \
  --long_context
```

Flags mirror those in the Weighted Aspect Recall script:

- **`--task`**, **`--k`**, **`--score_file`**, **`--output_dir`**, **`--cache_dir`**, **`--long_context`**, **`--all`**
- **`--alpha`**: novelty parameter for alpha-nDCG
- **`--save_json`**, **`--save_excel`**: optional outputs for summaries and Excel matrices

### Adding a New Retriever

To add a new retriever:

1. **Implement a retrieval function** in `retrievers.py` that:
   - Accepts `(queries, query_ids, documents, excluded_ids, instructions, doc_ids, task, cache_dir, long_context, model_id, checkpoint, **kwargs)`
   - Returns a mapping from query id to scores or ranked list of document ids
2. **Register it** in the `RETRIEVAL_FUNCS` dictionary in `retrievers.py` under a new key (e.g., `"my-retriever"`).
3. **Add configuration files** under `configs/<your-model>/<task>.json` with fields like `instructions` and `instructions_long`.
4. **Expose the model name** in the `choices` for `--model` in `run.py`.

### Project Structure

High-level layout:

- `run.py` – main script to run retrieval for a given task/model and write `score.json`
- `retrievers.py` – implementations of retrievers and helper utilities (embedding, scoring, caching)
- `configs/` – per-model, per-task configuration JSONs (instructions, parameters)
- `evaluation/` – evaluation scripts for Weighted Aspect Recall@K and weighted alpha-nDCG@K
- `cache/` – dataset and embedding caches (created automatically)
- `outputs/` – run directories containing `score.json` (and optional `results.json`)

### Citation

If you use this code or the BRIGHT dataset, please cite the corresponding paper.  
Replace the placeholder below with the official citation:

```text
@inproceedings{bright202X,
  title   = {BRIGHT: <Full Paper Title>},
  author  = {<Author List>},
  booktitle = {<Conference / Journal>},
  year    = {202X}
}
```

### License

Add your license information here (e.g., MIT, Apache-2.0), or link to a `LICENSE` file if present.


