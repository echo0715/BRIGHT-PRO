## BRIGTH-PRO

This repository contains code and data pipelines for the **Rethinking Evaluation of Reasoning-Intensive Retrieval:Toward Real-World Deep Research Workflows** paper. It focuses on retrieval settings where models must reason about **diverse, aspect-rich documents**, going beyond shallow lexical or embedding similarity.

- **Benchmarks & data**:
  - Construct `BRIGHT-PRO` dataset, with tasks such as biology, earth_science, economics, psychology, robotics, stackoverflow, and sustainable_living.
  - Documents are annotated with aspects and aspect weights, enabling **aspect-aware** related metrics.

- **Code structure**:
  - `retrieval/`: standalone scripts for running retrieval baselines (BM25, embedding models, API-based models, etc.) on BRIGHT-PRO, plus evaluation scripts (e.g., alpha-nDCG, weighted aspect recall).
  - `agentic_retrieval/`: search agents that interact with large language models (e.g., Azure OpenAI) to perform multi-step, agentic retrieval and produce answers; includes its own environment and evaluation scripts.

For environment setup and exact commands:

- See `retrieval/README.md`
- See `agentic_retrieval/README.md`


