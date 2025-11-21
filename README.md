## Reasoning-Intensive Retrieval (BRIGHT)

This repository contains code and data pipelines for the **Reasoning-Intensive Retrieval** paper (ARR 2025). It focuses on retrieval settings where models must reason about **diverse, aspect-rich documents** and support **long-context, multi-step queries**, going beyond shallow lexical or embedding similarity.

- **Benchmarks & data**:
  - Uses the BRIGHT family of datasets, including `ya-ir/BRIGHT-PRO`, with tasks such as biology, earth_science, economics, psychology, robotics, stackoverflow, and sustainable_living.
  - Documents are annotated with aspects and aspect weights, enabling **aspect-aware, diversified retrieval** metrics.

- **Code structure**:
  - `retrieval/`: standalone scripts for running classical and neural retrieval baselines (BM25, embedding models, API-based models, etc.) on BRIGHT-PRO, plus evaluation scripts (e.g., alpha-nDCG, weighted aspect recall).
  - `agentic_retrieval/`: search agents that interact with large language models (e.g., Azure OpenAI) to perform multi-step, agentic retrieval and produce answers; includes its own environment and evaluation scripts.

- **What this repo lets you do**:
  - Run **retrieval baselines** over BRIGHT tasks and cache embeddings / scores.
  - Run **agentic retrieval agents** that iteratively call search over BRIGHT, then generate answers.
  - Evaluate systems with **aspect-aware metrics** such as alpha-nDCG and weighted aspect recall.

For environment setup and exact commands:

- See `retrieval/README.md` for how to set up a minimal environment and run retrieval baselines.
- See `agentic_retrieval/README.md` for how to install via `uv`, configure Azure OpenAI, run search agents, and compute evaluation metrics.


