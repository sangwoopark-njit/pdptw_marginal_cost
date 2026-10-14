# Marginal-Cost Clustering for Courier Dispatch

Code and data for the paper *Marginal-Cost Clustering for Courier Dispatch in
Pickup and Delivery Problems with Time Windows and Fleet Sizing*.

The method decomposes a pickup-and-delivery problem with soft time windows and
a fleet-sizing decision into one assignment of requests to couriers followed by
exact routing of each courier. Requests are grouped by the **marginal cost**
each courier would incur to take them, rather than by spatial or
spatio-temporal distance. Each outer iteration builds candidate lists,
computes the marginal costs, solves one assignment integer program that also
decides which couriers to open, improves the result by relocate/swap descent,
and routes every dispatched courier exactly.

## Repository layout

```
code/                    all Python sources, run from this directory
  cluster_sparse.py      the proposed method (production version)
  routing.py             exact single-courier router (label setting with dominance)
  design.py, budget.py   instance generator and time-window preprocessing
  clustering_baselines.py  sweep, Fisher-Jaikumar, k-means, spatio-temporal
                         k-means, capacitated k-means, Ward agglomerative
  alns_baseline.py       ALNS reference (Ropke & Pisinger operator set)
  experiment_*.py, sweep_*.py, check_*.py, crossover.py   the experiments
  make_tables.py, crossover_table.py, table_style.py      results tables
  figure_*.py            figures
  results_*.csv          the raw results reported in the paper
  data/                  15 Li & Lim PDPTW instances
figures/                 the figures as they appear in the paper
tables/                  results section as generated LaTeX
```

## Setup

Python 3.12. Tested with numpy 2.2, scipy 1.13, pandas 2.3, matplotlib 3.10
and alns 7.0.

```
pip install -r requirements.txt
```

All experiments run single-threaded. The assignment integer program is solved
with HiGHS through `scipy.optimize.milp`. `highspy` and `ortools` are needed
only by an auxiliary baseline in `baselines.py`, which none of the paper's
experiments call.

## Reproducing the tables and figures from the stored results

These steps read the CSV files shipped in `code/` and take under a minute.

```
cd code
python make_tables.py          # writes ../tables/generated_results.tex
python figure_sensitivity.py   # Fig. 4, parameter sweeps
python figure_steps.py         # Fig. 2, one instance step by step (re-solves it, ~5 s)
python figure_paper.py         # Fig. 3, aligned vs misaligned (re-solves 10 instances, ~1 min)
```

Fig. 1 (the pipeline diagram) is TikZ: `figures/fig_pipeline.tex`.

The figures are typeset in XCharter to match the manuscript. Set the
environment variable `XCHARTER_OTF` to the path of `XCharter-Roman.otf`
(shipped with the TeX `xcharter` package). Without it, the figures fall back
to DejaVu Serif.

## Re-running the experiments

Run each command from `code/`. Each one writes the CSV named beside it.
`experiment_paper.py` **appends** to an existing file, so delete the old CSV
before re-running. Times are single-core wall-clock on the machine used for
the paper.

| Paper content | Command | Output | Time |
|---|---|---|---|
| Main comparison, φ = 100 | `python experiment_paper.py --part main` | `results_paper_main.csv` | ~20 min |
| Dispatch free, φ = 0 | `python experiment_paper.py --part main --phi 0 --out results_paper_main_phi0.csv` | `results_paper_main_phi0.csv` | ~22 min |
| Sweep of φ | `for p in 25 50 200 400; do python experiment_paper.py --part main --fleets 12 --phi $p --out results_paper_phisweep.csv; done` | `results_paper_phisweep.csv` | ~13 min |
| Misalignment sweep | `python experiment_paper.py --part main --fleets 12 --aligns 0,0.25,0.5,0.75,1 --seeds 8 --out results_paper_alignsweep.csv` | `results_paper_alignsweep.csv` | ~22 min |
| Fleet size | `python experiment_paper.py --part main --fleets 6,9,12,18 --seeds 4 --out results_paper_fleet.csv` | `results_paper_fleet.csv` | ~18 min |
| Li & Lim batches | `python experiment_paper.py --part lilim` | `results_paper_lilim.csv` | ~13 min |
| Scaling, \|R\| = 100–1000 | `python experiment_paper.py --part scale --fleets 30,60,120,300 --seeds 2` | `results_paper_scale.csv` | ~3 h |
| Time for ALNS to match | `python crossover.py` | `results_crossover.csv` | ~1 h |
| Penalty rate ρ | `python sweep_rho.py` | `results_paper_rho.csv` | ~22 min |
| Window width | `python sweep_window.py` | `results_paper_window.csv` | ~22 min |
| Delivery mixing | `python sweep_mixing.py` | `results_paper_mixing.csv` | ~9 min |
| Temporal weight ω | `python sweep_st_weight.py` | `results_paper_stweight.csv` | ~1 min |
| Slack-based rules | `python experiment_slack.py` | `results_paper_slack.csv` | ~2 min |
| Lagrangian assignment | `python experiment_lagrangian.py --fleets 12,30,60` | `results_paper_lagrangian.csv` | ~6 min |
| Interaction of marginal costs | `python check_interaction.py --seeds 4` | `results_paper_interaction.csv` | minutes |
| Restart determinism | `python check_determinism.py` | `results_paper_determinism.csv` | minutes |
| ν × \|K\| grid | `python sweep_nu_fleet.py` | `results_nu_fleet.csv` | hours |

For a quick check that the installation works, solve one instance (about 15 s):

```
python experiment_paper.py --part main --fleets 12 --aligns 1.0 --seeds 1 --alns_seeds 1 --out smoke.csv
```

The proposed method and all twelve clustering baselines are deterministic,
so this reproduces the corresponding rows of `results_paper_main.csv`
exactly. ALNS is stopped on wall-clock time, so its results vary with
machine speed and load.

Note: `results_crossover.csv` predates a change in `crossover.py` that now
also records each ALNS seed separately. Re-running it adds the
`alns_*_s0` / `alns_*_s1` columns. `crossover_table.py` reads either format.

## Data

`code/data/` holds 15 instances from the Li & Lim (2001) PDPTW benchmark (the
100-customer `lc1`, `lc2`, `lr1`, `lr2`, `lrc1` and `lrc2` families), from the
SINTEF set (mirror: <https://github.com/zhu-he/pdptw-data>). `instance.py`
adapts them as described in the paper. All other instances are generated
reproducibly by `design.py` from a seed.

> Li, H., Lim, A., 2001. A metaheuristic for the pickup and delivery problem
> with time windows. Proc. 13th IEEE International Conference on Tools with
> Artificial Intelligence, 160–167.
