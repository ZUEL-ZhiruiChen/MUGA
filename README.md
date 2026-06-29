# MVHGCL

This repository contains the implementation of **MVHGCL**, a multi-view hypergraph representation learning framework for node classification. The model constructs multiple complementary hypergraph views and fuses them through a node-level reliability gating mechanism.

## Framework

MVHGCL mainly contains four parts:

1. **Multi-view hypergraph construction**
   - Structural hypergraph view from graph topology.
   - Feature-neighbor hypergraph view from feature-space nearest neighbors.
   - Prototype hypergraph view from semantic prototype assignments.

2. **View-specific hypergraph encoding**
   - Each view is encoded by an independent hypergraph encoder.
   - The encoders learn view-dependent node representations.

3. **Node-level reliability gating**
   - A gating module estimates the reliability of each view for each node.
   - View representations are adaptively fused into the final node embedding.

4. **Auxiliary self-supervised objectives**
   - Cross-view learning and elastic distance constraints are used to improve representation stability.
   - The final node classifier is trained together with the selected auxiliary objectives.

## Repository Structure

```text
mvhgcl/
+-- configs/                 # Dataset and training configuration files
+-- models/                  # MVHGCL model, hypergraph layers, and losses
+-- utils/                   # Data loading and utility functions
+-- scripts/                 # Helper scripts
+-- run_node.py              # Main entry for a single experiment
+-- run_selected_combos.py   # Batch runner for selected datasets and variants
+-- run_selected_combos.bat  # Windows batch shortcut
```

The main training pipeline is in `run_node.py`. The supplementary scripts in `appendix_experiments/` are used for additional analysis and are not required for basic model training.

## Environment

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the basic dependencies:

```powershell
python -m pip install -U pip
pip install numpy scipy scikit-learn torch pyyaml
```

## Quick Start

Run a minimal baseline experiment:

```powershell
python run_node.py --config configs/node/chameleon_filtered.yaml --variant ce_only --seeds 0 --output-dir test_runs
```

Run the full MVHGCL variant:

```powershell
python run_node.py --config configs/node/chameleon_filtered.yaml --variant soft_hg+dominant_distill+discrim_elastic --seeds 0 --output-dir test_runs
```

Run selected experiments in batch:

```powershell
python run_selected_combos.py --skip-bootstrap --seeds 0
```

## Common Variants

```text
ce_only
soft_hg
dominant_distill
discrim_elastic
soft_hg+dominant_distill+discrim_elastic
```

Variants can be combined with `+`. For example, `soft_hg+discrim_elastic` enables both soft hypergraph construction and the elastic distance constraint.

## Outputs

For a single run, results are saved under:

```text
<output-dir>/<dataset>/<variant>/
```

For batch runs, aggregate results are usually saved under:

```text
combo_runs/search/
+-- aggregate.csv
+-- aggregate.json
`-- aggregate.md
```

## Datasets

Dataset configuration files are located in:

```text
configs/node/
```

Commonly used datasets include `chameleon_filtered`, `squirrel_filtered`, `texas`, and `tolokers`. Additional configurations such as `actor`, `coauthor_cs`, `coauthor_physics`, and `questions` are also provided.

## Citation

If this repository is useful for your research, please cite the corresponding MVHGCL paper.
