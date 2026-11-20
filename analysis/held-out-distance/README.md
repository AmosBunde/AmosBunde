# Error against distance from the training data

**Does a model's error grow with distance from its training data, and does it grow the same way for every kind of model?**

Amos Bunde, November 2026

I had run three unrelated studies that each compared an easy test with a hard one: crop-type maps, novel-view 3D reconstruction, and runtime prediction for neural networks. Each reported a gap. This repository goes back to the per-item results of all three and asks a sharper question. Instead of two test designs, it places every test item at its actual distance from the training data and draws the curve.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/distance-dark.svg">
  <img src="docs/img/distance-light.svg" alt="Three panels. Crop maps, in macro-F1: the temporal CNN falls from 0.32 to 0.24 with distance and the forest on the raw series from 0.22 to 0.16, while the forest on growth-curve parameters stays near 0.16 throughout. The other two panels show error relative to the test items nearest the training data. Novel-view 3D: image error rises to 2.15 times while depth error stays near 1.1 times. Runtime prediction: gradient boosting rises to 3.45 times and log-linear regression to 2.25 times, while the mechanistic simulator stays near 0.93 times." width="100%">
</picture>

Shaded bands are 95% intervals from a cluster bootstrap. The crop panel shows macro-F1 itself, so each model's level is visible as well as its slope; the other two panels show error relative to the test items nearest the training data.

## What it shows

| Study | Distance measured as | Model output | Error far from training, relative to near (95% interval) |
|---|---|---|---|
| Crop-type maps | metres to the nearest training field | temporal CNN on the raw series | 1.12 (1.07 to 1.16) |
| | | forest on the raw series | 1.08 (1.03 to 1.13) |
| | | forest on growth-curve parameters | 1.02 (0.99 to 1.05) |
| Novel-view 3D | degrees to the nearest training camera | appearance: image error | 2.15 (1.59 to 2.89) |
| | | geometry: depth error | 1.11 (1.04 to 1.17) |
| Runtime prediction | test workload size ÷ largest training workload | gradient boosting | 3.45 (2.82 to 4.10) |
| | | log-linear regression | 2.25 (1.90 to 2.69) |
| | | mechanistic simulator | 0.93 (0.71 to 1.16) |

Three things stand out.

1. **Error rose with distance in every fitted predictor.** For crop maps, macro-F1 fell from 0.316 for fields within 50 m of a training field to 0.238 for fields more than 800 m away, across six distance groups without a reversal. For 3D views, PSNR fell from 32.7 dB at about 10 degrees to 29.7 dB beyond 26 degrees. For gradient boosting, runtime error went from 24% inside the training range to 82% on workloads more than 30 times larger.
2. **Two outputs did not follow.** The depth recovered by the radiance field and the runtime predicted by the mechanistic simulator stayed close to their near-training error at every distance. The simulator's interval includes 1.0; depth rose 11%.
3. **Those two are the outputs constrained by a model of the problem.** Depth comes from geometry that many views must agree on. The simulator adds up operator costs it measured separately. The outputs that degraded were fitted directly to the target: colour, crop class, a regression on workload features.

That third point began as an interpretation of two cases found after the fact. It makes a prediction that could fail: a predictor that encodes how its domain works should show a flatter curve than one that does not, on the same test items. The crop study had no such model, so I wrote the prediction down and built one.

## The prediction, tested

The plan is in the crop study's [FOLLOW_UP_PLAN.md](https://github.com/AmosBunde/crop-map-spatial-validation/blob/main/FOLLOW_UP_PLAN.md), committed before the new model was run on any labelled field. The new model describes each field by 18 parameters of a growth curve (base, amplitude, peak day, rise, fall and fit error, for three vegetation indices) and trains the same random forest on them that the baseline trains on the raw series. The two differ only in what the forest is given.

`R` is the error in the farthest distance group over the error in the nearest.

| Rule set in advance | Result | Outcome |
|---|---|---|
| Guard: the new model must have learned something (macro-F1 near the training data at least 0.153, halfway between the majority baseline and the raw-series forest) | 0.171 | passed |
| P1: `R` lower than the raw-series forest's, interval excluding zero | 1.02 against 1.08; difference −0.059 (−0.101 to −0.018) | supported |
| P2: `R` lower than the network's | 1.02 against 1.12; difference −0.097 (−0.142 to −0.048) | supported |
| P3: far from the training data, macro-F1 not below the raw-series forest's | 0.156 against 0.157; difference −0.001 (−0.017 to 0.016) | supported |

I wrote in the plan that I expected P1 to fail. It held.

**What it does and does not mean.** The structured model is not a better crop classifier. Near the training data it is clearly worse than the raw-series forest (0.171 against 0.218), and the network beats both at every distance (0.238 at the far end). What the test shows is narrower and, I think, more useful: all of the raw-series forest's advantage over the structured one sits within a few hundred metres of a training field. Beyond 400 m the two are level. The extra accuracy that the raw series buys is tied to the place it was learned in. A flat curve can come from carrying signal that transfers or from discarding signal that does not, and this result is the second kind.

## Why this is more than the original gaps

The original studies compared designs: random split against block split, scattered views against a held-out patch. A gap between two designs can come from anything that differs between them. Here each item carries its own distance, items from different designs share a group when their distances match, and the curve is checked for a trend. For crops this needed the folds, which the original study did not store. They are deterministic, so `scripts/extract_crop.py` rebuilds them with the study's own code and measures each test field's distance to its nearest training field.

## How far to trust it

- **The distance curves are a re-analysis decided after the three studies were finished.** Only the phenology test was planned in advance, and that plan was written the same day by the person who ran it and was not registered anywhere. I looked at a first version of the crop curve, pooled over every design, before settling on the size-matched version and its coarser distance groups, so treat every number as exploratory.
- **The crop curve is shallow.** The network's relative error rose 12%, because error is high everywhere (macro-F1 is about 0.3 on seven classes). The trend is steady, and all 2,000 bootstrap resamples put the farthest group above the nearest, but the size is small.
- **The phenology test is one model in one region.** The growth curve is a bell fitted to 13 dates between June and November, on intercropped smallholder fields. A different structured model, or a different season, could give a different answer. The guard passed, but not by much (0.171 against a bar of 0.153).
- **Crop distance is confounded with training-set size unless handled.** The buffered and tile designs train on fewer fields. The headline curve uses only the random and 2 km block designs, which train on the same number of fields. Pooling every design gives 1.14 (1.09 to 1.18), in `results/summary.json` under `crop_all_designs`.
- **Class mix varies a little between crop distance groups**, and macro-F1 is sensitive to rare classes. Accuracy moved less (0.454 to 0.427).
- **The 3D scenes are synthetic** and the far groups are small (22 to 45 test views each, from 18 trained models).
- **The runtime measurements are noisy.** Repeat timings in that study differed by about 30%. The contrast here (3.45 against 0.93) is larger than that; small differences are not. The "inside the training range" group comes from a different train/test split than the others. Inside the size split alone, gradient boosting's error still rose 3.15 times from the nearest group to the farthest, and the simulator's 1.05 times.
- **The three distances are different quantities.** The panels share a vertical scale but not a horizontal one. Nothing here says that 30 degrees of viewing angle is "as far" as 1 km of farmland.
- **Reward gaming is not included.** That study held out a type of exploit, which has no distance.
- **Related work exists.** Distance-aware evaluation is established in spatial statistics, and the idea that structured models extrapolate better than fitted ones is old. I have not surveyed either literature. What is here is a measurement across three domains with shared code.

## Reproduce it

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). The extracted per-item data is committed, so the analysis runs from a fresh clone in about four minutes.

```bash
git clone https://github.com/AmosBunde/AmosBunde
cd AmosBunde/analysis/held-out-distance
uv sync
uv run pytest -q             # 11 tests
uv run heldout summarise     # about 4 minutes; writes results/summary.json
uv run heldout figures       # writes docs/img/
```

To rebuild the extracts from the original studies, clone them, complete their own setup steps, and run `scripts/extract_crop.py` inside the crop study's environment and `scripts/extract_views_and_runtime.py` here. Both scripts say at the top where they expect the other repositories to be.

## Layout

| Path | What it holds |
|---|---|
| `data/` | Per-item extracts: field predictions from three crop models, 432 test views, 1,728 runtime cells |
| `scripts/` | The two extraction scripts |
| `src/heldout_distance/analysis.py` | Grouping by distance, error per group, cluster bootstrap |
| `src/heldout_distance/stats.py` | Macro-F1, cluster bootstrap, rank correlation |
| `src/heldout_distance/figures.py` | The figure, in light and dark versions |
| `results/summary.json` | Every number in this README |
| `tests/` | Statistics against known values, a planted-trend check, shape checks on the data |

## Source studies

- [crop-map-spatial-validation](https://github.com/AmosBunde/crop-map-spatial-validation)
- [Novel-View_3D-reconstruction-evaluation](https://github.com/AmosBunde/Novel-View_3D-reconstruction-evaluation)
- [Tensor-Workload_runtime-simulator](https://github.com/AmosBunde/Tensor-Workload_runtime-simulator)

## What I would do next

1. Build a structured crop model that is also competitive near the training data, for example by giving the network the growth-curve parameters alongside the series, and see whether its curve stays flat.
2. Fit the curves rather than describe them, and ask whether the near-training part of a curve predicts the far part. If it does, the gap could be estimated before any far data is collected.
3. Repeat the 3D and runtime parts on real captures and on an accelerator.

## Licence

MIT. See [LICENSE](LICENSE).
