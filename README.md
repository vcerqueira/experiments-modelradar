# ModelRadar - Experiments

This repository contains the experiments for the paper "ModelRadar: Aspect-based Forecasting Accuracy" (Cerqueira et al., 2025).
[ModelRadar](https://github.com/vcerqueira/modelradar) is a framework for evaluating forecasting performance across different dimensions of interest. 

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- modelradar
- Nixtla ecosystem (check specific versions in requirements.txt)
- Install dependencies listed in `requirements.txt` using:

```bash
pip install -r requirements.txt
```

## Running Experiments

To reproduce the experiments from the paper:

1. Execute the experimental scripts in the folder scripts/experiments/run. For example:

```bash

# Prepare metadata and features from time series datasets
python scripts/experiments/run/features.py
python scripts/experiments/run/metadata.py
python scripts/experiments/run/sf_anomaly.py
# Hyperparameter optimization for neural forecasting models
python scripts/experiments/run/nf_hypertuning.py
# Run neural forecasting models after optimization
python scripts/experiments/run/nf_from_config.py
# Running machine learning regression-based forecasting models
python scripts/experiments/run/mlf_plus_hpo.py
# Running classical forecasting models
python scripts/experiments/run/sf.py
# Running neural-based seed robustness analysis
python scripts/experiments/run/nf_robustness.py

```

2. Analyze the results with the scripts on folder scripts/experiments/analysis. For example:
```bash
# Combine the results of different types of methods
python scripts/experiments/analysis/1_consolidate_experiments.py
# Select a subset of models for aspect-based analysis
python scripts/experiments/analysis/2_pre-analysis.py
# Running ModelRadar on a subset of models
python scripts/experiments/analysis/3_analysis.py
```

## Results overview

Here's an overview of the forecast accuracy (SMAPE) of several models across different dimensions:

Accuracy scores over several dimensions 
![multidim_performance](assets/screenshots/spider.png)

Controlling accuracy by anomaly status
![anomalies](assets/screenshots/on_anomalies.png)

You can find several other plots in the folder *scripts/experiments/outputs*

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cerqueira2025modelradar,
  title={ModelRadar: Aspect-based Forecast Evaluation},
  author={Cerqueira, Vitor and Roque, Luis and Soares, Carlos},
  journal={arXiv preprint arXiv:2504.00059},
  year={2025}
}
```

## Contact

For questions or feedback about this implementation, please open an issue in this repository.
