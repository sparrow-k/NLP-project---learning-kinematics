# The Kinematics of Learning

NLP final project, Tel Aviv University: Dror Kernan, Roee Lazarovich.
The paper is `report/report.pdf`.

## Repository structure

| Path | Purpose |
|---|---|
| `*.py`, `run_matrix.sh` | Research code (see below) |
| `data/` | Phase 1 measurements per model (`kinematics_*.parquet`) and dataset metadata |
| `results/` | Phase 1 audits and hypothesis tests; Phase 2 summaries and statistics |
| `runs/` | One folder per Phase 2 training run (config, logs, evaluations, summary) and the overfit sanity check |
| `report/` | Paper source (`report.tex`, `phase2_results.tex`), generated numbers and table, figures, bibliography, compiled PDF |
| `requirements.txt`, `requirements.lock.txt` | Python dependencies (loose / exact versions used) |
| `NLP_project_training_kinematics.pdf` | Original project proposal |

## Scripts

| File | Purpose |
|---|---|
| `config.py` | Shared settings: models, the 29 checkpoint steps, matrix names, paths |
| `smoke_test.py` | Checks the Pythia checkpoint layout and the fused Q/K/V split |
| `kinematics.py` | Loads a checkpoint, splits Q/K/V per head, computes cosine and displacement |
| `extract_kinematics.py` | Runs `kinematics.py` over all checkpoints of one model → `data/kinematics_<model>.parquet` |
| `compute_stabilization.py` | Settling times (t50, t90, thresholds), data audit and hypothesis tests → `results/` |
| `hypotheses.py` | The statistical tests for H1–H5 (used by `compute_stabilization.py`) |
| `plot_kinematics.py` | Phase 1 analysis figures → `figs/` |
| `plot_depth_summary.py` | Paper Figure 1 (depth trend per matrix type) |
| `data_prep.py` | Downloads and tokenizes WikiText-103 → `data/` |
| `schedules.py` | Turns Phase 1 settling times into switch steps and per-arm learning-rate factors |
| `train.py` | Trains one run (one arm, one seed); also `--overfit` and `--smoke` sanity checks |
| `run_matrix.sh` | Runs all arms × seeds one at a time, skipping finished runs |
| `plot_phase2.py` | Collects all runs, paired tests → `results/phase2_*` and analysis figures |
| `plot_paper_figs.py` | Paper Figures 2 and 3 |
| `make_report_numbers.py` | Writes every number in the paper to `report/numbers.tex`, and Table 3 |
| `report/build.sh` | Compiles the paper (pdflatex + bibtex) |

## Order in which the code was run

1. `python smoke_test.py` — confirm the checkpoint layout and the Q/K/V split.
2. `python extract_kinematics.py --model EleutherAI/pythia-70m` (then `pythia-160m`) — measure every matrix at every checkpoint.
3. `python compute_stabilization.py --model EleutherAI/pythia-70m` (then `pythia-160m`) — settling times, audit, hypothesis tests.
4. `python plot_kinematics.py --all` — Phase 1 figures.
5. `python data_prep.py` — tokenize WikiText-103.
6. `python train.py --overfit` and `python train.py --smoke` — sanity checks before the real runs.
7. `sh run_matrix.sh 4000` — the first 24 Phase 2 runs (baseline, attn-first, attn-second, alternating, attn-freeze).
8. `python plot_phase2.py --steps 4000` — aggregate and test the Phase 2 runs.
9. `python extract_kinematics.py --model EleutherAI/pythia-410m` and `python compute_stabilization.py --model EleutherAI/pythia-410m` — add the third model size.
10. `python plot_kinematics.py --all` and `python plot_depth_summary.py` — Phase 1 figures with all three models.
11. `sh run_matrix.sh 4000` — the 12 learning-rate-matched runs (`attn-first-lrm`, `attn-second-lrm`); finished runs are skipped.
12. `python plot_phase2.py --steps 4000` and `python plot_paper_figs.py` — final Phase 2 statistics and paper figures.
13. `python make_report_numbers.py` — write the paper's numbers and Table 3.
14. `sh report/build.sh` — compile `report/report.pdf`.
