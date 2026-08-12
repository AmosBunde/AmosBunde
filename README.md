# Amos Bunde

**Data and AI Infrastructure Engineer | Production LLM Systems, Retrieval, and Evaluation**

I build data platforms and LLM systems that survive contact with production: clinical decision support used by clinicians, evaluation harnesses that catch silent failures before they ship, and the streaming pipelines and Kubernetes infrastructure underneath them. Twelve years across data engineering, MLOps, and cloud infrastructure on AWS, GCP, and Azure. MSc from Liverpool John Moores University, where my research applied reinforcement learning and Bayesian optimization to HPC workload scheduling. HPC certified at HLRS Stuttgart (MPI/OpenMP, CUDA).

**Currently open to applied AI and ML infrastructure roles.** Based in Nairobi, open to relocation.

---

## Selected work

- **Afya Gemma**, a clinical RAG decision-support system (MedGemma on Vertex AI, hybrid dense plus BM25 retrieval with reranking, deterministic safety gates for dosing content). Winner of a Google GenAI Accelerator Award. The public evaluation methodology lives in the repository below.

- **[grounding-conditional-eval](https://github.com/AmosBunde/grounding-conditional-eval)**. GC-Accuracy, an evaluation metric that separates honest abstention from failure in retrieval-augmented systems. Standard accuracy punishes a model for refusing to answer when retrieval fails; GC-Accuracy scores it correctly. Includes a technical report and a worked example where the same retrieval fix scores as a regression under standard accuracy and as an improvement under GC-Accuracy. CC BY 4.0.

- **[corpus-gate](https://github.com/AmosBunde/corpus-gate)**. Contract-intelligence RAG with a fine-tuned agent and a graded evaluation gate: no deploy without passing the eval suite. Built on public commercial contracts (CUAD, SEC EDGAR exhibit 10 filings). Open-weights LLM, vector retrieval, agentic tool calling, FastAPI and React. Active build with a one issue, one branch, one PR workflow.

- **[anomaly-desk](https://github.com/AmosBunde/anomaly-desk)**. Agentic incident triage with an LLM-as-judge harness. Streaming Kafka ingest, typed multi-agent workflow with mandatory citations, versioned escalation policy, dual scoreboard (judge score against operator override rate), OpenTelemetry tracing with cost per triage, and an eval-regression deploy gate on Kubernetes. Active build.

- **[loss-curve](https://github.com/AmosBunde/loss-curve)**. A small language model training study: hand-written transformer, byte-level BPE tokenizer, and training loop; an evaluation suite frozen before any experiment runs; one-variable experiments with two-seed reruns; and a deliberate breakage catalog. The deliverable is the written experiment log, not the checkpoints. Active build.

- **[Agent-Arena](https://github.com/AmosBunde/Agent-Arena)**. An open-source LLM agent evaluation platform: cost per correct answer as the headline metric, provider-agnostic adapters, and content-addressed trace storage for deterministic replay.

## Open source

- **EleutherAI/lm-evaluation-harness**: diagnosed and fixed a silent prompt-rendering bug in the AfriXNLI benchmark tasks. Single-brace format strings passed through Jinja2 unrendered, so roughly 35 task configurations across 18 African languages were evaluating models on prompts that contained no premise and no hypothesis, while raising no error and returning plausible scores. Proved the failure with prompt-hash analysis (200 documents rendered 1 distinct prompt before the fix and 200 distinct prompts after) and authored the generator-level fix. [Issue #3930](https://github.com/EleutherAI/lm-evaluation-harness/issues/3930) · [PR #3932](https://github.com/EleutherAI/lm-evaluation-harness/pull/3932)

## Stack

- **Data platform**: Kafka, Flink, Airflow, Dagster, Elasticsearch, PostgreSQL, MySQL, MongoDB, SQL Server
- **ML and evaluation**: PyTorch, TensorFlow, scikit-learn, RAG pipelines (ChromaDB, hybrid retrieval, reranking), lm-evaluation-harness, Langfuse
- **Infrastructure**: Kubernetes, Docker, Terraform, Ansible, GitHub Actions, Jenkins, Grafana, across AWS, GCP, and Azure
- **Languages**: Python, SQL, Scala, Java

## Elsewhere

- Portfolio: [iam-amos.io](https://iam-amos.io)
- LinkedIn: [in/amos-bunde](https://www.linkedin.com/in/amos-bunde)


