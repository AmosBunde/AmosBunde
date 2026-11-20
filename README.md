<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner-light.svg" alt="Amos Bunde. How wrong is a test set that sits next to its training data?" width="100%">
</picture>

A model is usually scored on test data drawn from the same place as its training data: neighbouring fields, neighbouring camera positions, workloads of the same size. Deployment asks for something else. I measure the difference, and I have now run the same experiment in four unrelated domains.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/one-question-dark.svg">
  <img src="assets/one-question-light.svg" alt="Four panels, one per study. Crop-type maps: macro-F1 falls from 0.29 to 0.21 when a whole image tile is held out. Novel-view 3D: PSNR falls from 32.9 to 31.0 dB when a whole patch of viewpoints is held out. Runtime prediction: gradient boosting error rises from 23.6% to 50.2% on larger workloads while a mechanistic simulator stays near 20%. Reward gaming: agents trained to game graders game an unseen exploit type in 20% of episodes, agents trained honestly in 0%." width="100%">
</picture>

<sub>Each panel has its own vertical scale; read the numbers, not the slopes. All four are small studies on modest hardware, and each report lists its limits. The runtime measurements were noisy (30% repeat error).</sub>

**A pattern I see, not yet a result.** The gap is not a property of the model alone; it depends on how far the test asks the model to travel. Where the model carried structure about the problem (a simulator built from operator costs, the geometry inside a radiance field), the gap was small. Where it had only fitted a surface, it was large. Four small studies do not establish that, and I cannot yet predict the gap before collecting the held-out data. That is the question I most want to answer.

I am Head of Data and AI at Inkomoko in Nairobi, with twelve years in data, machine learning and platform engineering. [Portfolio](https://iam-amos.io) | [LinkedIn](https://www.linkedin.com/in/amos-bunde) | [ORCID](https://orcid.org/0009-0003-3419-5715)

## The studies

Each is small and complete: a plan written before the experiments ran, tests, committed result files, and a `REPORT.md` that says which predictions failed.

| Question | What I found | Built with |
|---|---|---|
| [How much does a random split overstate crop-map accuracy?](https://github.com/AmosBunde/crop-map-spatial-validation) | Macro-F1 fell from 0.29 to 0.21 when a whole image tile was held out instead of random fields (3,286 fields, Busia County, Kenya). [Interactive write-up](https://iam-amos.io/work/crop-map-spatial-validation/). | PyTorch, PostGIS, rasterio |
| [Does the usual held-out-view test flatter a 3D reconstruction?](https://github.com/AmosBunde/Novel-View_3D-reconstruction-evaluation) | Holding out a whole patch of viewing directions cost 1.9 dB of PSNR, while depth error rose only 3%. Radiance field and structure from motion written from scratch. | PyTorch, OpenCV |
| [If an agent learns to game one grader, does it game another?](https://github.com/AmosBunde/Reward-Gaming_RL-safety-environments) | Agents trained to game three exploit types gamed an unseen fourth in 20% of episodes. Agents trained without exploits never did. | PyTorch, PPO from scratch |
| [Real regression, or a broken measurement?](https://github.com/AmosBunde/Eval-Fleet_training-run-monitor) | A monitor for evaluations during training caught all 11 regressions of 5 points or more, with false alerts on 3 of 18 clean runs against 17 of 18 for a simple drop rule. | PyTorch, SciPy |
| [Can you predict a model's runtime without running it?](https://github.com/AmosBunde/Tensor-Workload_runtime-simulator) | A trace-driven simulator held about 20% error on workloads larger than any it was fitted on, where gradient boosting rose to 50%. Timing noise was high (30% repeat error) and is reported. | PyTorch, CPU only |
| [Does standard accuracy punish honest abstention?](https://github.com/AmosBunde/grounding-conditional-eval) | One retrieval fix read as a 4.5-point regression under standard accuracy and an 8.5-point gain under GC-Accuracy. Technical report and open benchmark. | Python |

### Predictions I got wrong

I keep these in the reports, because a write-up with no failed predictions usually means the predictions were written afterwards.

- I expected a held-out viewpoint to damage geometry more than appearance. It was the reverse: 3% against 56%.
- I expected agents to keep gaming a grader after the exploit became more costly than the task. They stopped as soon as it did.
- I expected paired item-level tests to detect regressions earlier than unpaired ones. They did not, at matched false-alarm rates.
- I expected a simulator calibrated only on microbenchmarks to beat a regressor on an unseen thread count. It lost.

## Open questions

- Can the gap between a comfortable test and a hard one be predicted from the training set alone?
- In the 3D study the held-out views lost image quality but kept their geometry, and view-dependent colour explained only a tenth of the loss. What explains the rest?
- Agents that learned to game graders went straight for an unfamiliar object in 88% of episodes but succeeded in gaming with it only 20% of the time. Does the habit survive when honest work is also rewarded during training?
- A simulator calibrated on a busy machine was biased one way and on a quiet machine the other. How should a performance model carry the state of the machine it was calibrated on?

## Background

<details>
<summary><b>Production systems I have led</b></summary>

- **Afya Gemma.** Clinical decision support grounded in Kenyan Ministry of Health guidelines, built at Aga Khan University Hospital and used by resident and intern doctors. Two-stage retrieval, release-blocking evaluation in CI, and deterministic safety gates for dosing content. Google GenAI Accelerator Award. The evaluation method is public in [grounding-conditional-eval](https://github.com/AmosBunde/grounding-conditional-eval).
- **Research and clinical data platform, Aga Khan University.** Lakehouse, streaming and governance across Kenya and Pakistan, including a real-time wearable stream from 615 health care workers for an NIH-funded consortium.
- **Commercial data platform, Copia Global.** About 250,000 transactions a day at 99.95% uptime, and a learning-to-rank model behind a live API that cut referral payouts by 60%.

Case studies with architecture diagrams are on [iam-amos.io/projects](https://iam-amos.io/projects/).

</details>

<details>
<summary><b>Open source</b></summary>

- **EleutherAI/lm-evaluation-harness.** Diagnosed a silent defect that left 35 AfriXNLI task configurations in 18 African languages scoring one identical prompt: the format strings passed through the template engine unrendered, with no error and plausible scores. I showed it with prompt hashes (200 documents produced 1 distinct prompt) and proposed a patch. [Issue #3930](https://github.com/EleutherAI/lm-evaluation-harness/issues/3930), [proposed patch #3932](https://github.com/EleutherAI/lm-evaluation-harness/pull/3932) (closed without merge).
- **[Agent-Arena](https://github.com/AmosBunde/Agent-Arena).** An evaluation platform for LLM agents: cost per correct answer as the headline metric and deterministic replay of traces.
- **[corpus-gate](https://github.com/AmosBunde/corpus-gate)** and **[anomaly-desk](https://github.com/AmosBunde/anomaly-desk).** Two builds in progress on one rule: nothing deploys without passing its evaluation suite.

</details>

<details>
<summary><b>Writing</b></summary>

- [Your Eval Harness Is Optimizing a Number That Hides the Thing You Care About](https://www.linkedin.com/pulse/your-eval-harness-optimizing-number-hides-thing-you-care-amos-b--3ourf/)
- [Our Clinical RAG Evaluation Was Rewarding the Wrong Behaviour](https://www.linkedin.com/pulse/our-clinical-rag-evaluation-rewarding-wrong-behaviour-amos-b--kkv7f/)
- [Static vs Dynamic Quantization: What Actually Breaks, and When](https://www.linkedin.com/pulse/static-dynamic-quantization-what-actually-breaks-when-amos-b--w6qsf/)

More on [iam-amos.io/writing](https://iam-amos.io/writing/).

</details>

<details>
<summary><b>Learning log: what I studied and what I made with it</b></summary>

What I studied, and what I made with it.

| When | What | What came out of it |
|---|---|---|
| Nov 2026 | 3D vision, reinforcement learning, performance modelling | The four newest studies above |
| 2026 | Earth observation and spatial statistics, self-taught | The crop-map study, including recovering coordinates for a dataset published without them |
| 2026 | MSc, Liverpool John Moores University | Dissertation on reinforcement learning and Bayesian optimisation for HPC workload scheduling: 12.3% less energy than first-come-first-served |
| 2026 | Udacity Nanodegree, Machine Learning Model Optimization | [UdaciMed](https://github.com/AmosBunde/UdaciMed): a chest X-ray classifier with 87% fewer parameters, CPU latency from 33 ms to 0.5 ms, sensitivity held at 98.2% |
| 2026 | Udacity Nanodegree, Responsible AI | A gender-bias audit of a fine-tuned language model, and its repair |
| 2026 | Stanford University, Agentic AI; BlueDot Impact, Technical AI Safety | Shaped the reward-gaming and evaluation-monitoring studies |
| Earlier | HLRS Stuttgart: MPI, OpenMP and CUDA | The systems background behind the runtime simulator |

**In progress, no empirical results yet:** [non-separable-uq](https://github.com/AmosBunde/non-separable-uq), a calibration statistic for preference models across social contexts, and [ubuntu-rl](https://github.com/AmosBunde/ubuntu-rl), preference optimisation over a stakeholder graph.

</details>

## How I work

- The plan, the splits and the metrics are written down before the full run.
- Every number in a README comes from a committed result file.
- Failed predictions and noisy measurements go in the report, with the reason if I know it.
- One issue, one branch, one pull request.

**Tools I use most:** Python, PyTorch, SQL, PostgreSQL and PostGIS, Kafka, Spark, Airflow, Kubernetes, Terraform, on Azure, AWS and Google Cloud.
