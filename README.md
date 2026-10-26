# Beyond Commit Counts: Detecting Unusual Developer Workflow Fragmentation from Public GitHub Activity

<p align="center">
  <strong>An Exploratory Machine Learning Study of Repository-Switching Patterns</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/ML-Isolation%20Forest-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Data-GH%20Archive-green" alt="GH Archive">
  <img src="https://img.shields.io/badge/Paper-IEEE-red" alt="IEEE Paper">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-purple" alt="Research Prototype">
</p>

## Overview

Developer activity is often summarized using simple measures such as event counts, commits, or the number of repositories involved. However, these measures do not fully describe the order in which activity occurs across repositories.

This project investigates whether **repository-switching patterns provide additional information for identifying unusual public GitHub activity**.

Using public GitHub event data, the study extracts activity-based and sequence-derived features and compares their behaviour under the unsupervised **Isolation Forest** algorithm.

The project is an exploratory software repository mining study. Its purpose is to understand how different feature representations affect anomaly flags, rather than to infer developers' productivity, mental state, or working conditions.

## Research Question

**Do sequence-derived repository-switching features change which public GitHub activity records are flagged as unusual compared with a conventional activity-based baseline?**

### Objectives

* Examine activity distributed across multiple repositories.
* Measure repository transitions and short-horizon repository re-entry.
* Compare activity-based, temporal-only, and combined feature configurations.
* Investigate the sensitivity of anomaly flags to model parameters.
* Explore whether the selected records differ across feature configurations.

## Methodology

The research pipeline consists of the following stages:

1. **Data collection:** Obtain selected public GitHub event archives from GH Archive.
2. **Preprocessing:** Parse relevant event types, clean records, and remove duplicate events according to the project's preprocessing procedure.
3. **Actor-week aggregation:** Group observed events by actor and week.
4. **Feature engineering:** Calculate activity volume, repository diversity, repository transitions, transition intensity, and short-horizon re-entry.
5. **Unsupervised modelling:** Apply Isolation Forest to three feature configurations.
6. **Comparison:** Examine flag counts, overlap between flagged sets, feature relationships, and parameter sensitivity.
7. **Interpretation:** Discuss the findings alongside sampling limitations and the absence of independently verified anomaly labels.

## Feature Engineering

| Feature                            | Description                                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Event count                        | Number of observed events associated with an actor-week                                                             |
| Distinct repositories              | Number of different repositories observed                                                                           |
| Normalized repository entropy      | Measures how activity is distributed across repositories                                                            |
| Repository transitions             | Number of changes between consecutive observed repositories                                                         |
| Transition intensity               | Repository transitions divided by the observed time span, with a minimum one-hour denominator                       |
| Short-horizon re-entry rate (SHRR) | Proportion of repository transitions that return to a previously observed repository within a specified time window |

These features describe public event sequences. They do not reveal why a repository switch occurred or whether it interrupted a developer's work.

## Dataset

The experiment used four selected GH Archive snapshots from **October 1, 2026**, corresponding to 00:00, 06:00, 12:00, and 18:00 UTC.

| Dataset characteristic                              |   Count |
| --------------------------------------------------- | ------: |
| Raw event lines                                     | 329,310 |
| Relevant events before deduplication and cleanup    | 312,125 |
| Clean unique events                                 | 311,034 |
| Distinct actors                                     | 108,666 |
| Distinct repositories                               | 158,960 |
| Actor-week records before filtering                 | 108,666 |
| Actor-week records retained for the main comparison |  31,500 |

The main comparison retained actor-week records containing at least two observed events.

**Data limitation:** These four selected hourly snapshots do not represent continuous GitHub activity across the full day or week. Automated accounts were not comprehensively filtered, and the sample should be treated as sparse observational data.

Dataset source: [GH Archive](https://www.gharchive.org/)

## Experimental Results

### Model Comparison

The three Isolation Forest configurations were evaluated on the same 31,500 filtered actor-week records using a nominal contamination setting of 0.05.

| Configuration     | Features                                                          | Flagged records | Flag rate |
| ----------------- | ----------------------------------------------------------------- | --------------: | --------: |
| Activity baseline | Event count, distinct repositories, normalized repository entropy |           1,555 |     4.94% |
| Temporal-only     | Repository transitions, transition intensity, SHRR                |           1,550 |     4.92% |
| Combined          | All six features                                                  |           1,570 |     4.98% |

Although the flag rates were similar, the sets of flagged records overlapped only partially. The Jaccard similarity was 33.55% between the activity baseline and temporal-only configuration, 54.55% between the baseline and combined configuration, and 66.49% between the temporal-only and combined configurations.

These results show that feature selection changes which records the model identifies as unusual. They do not establish that one configuration is more accurate.

### Sensitivity Analysis

The experiment also examined different contamination settings and short-horizon re-entry windows.

* Increasing the contamination setting increased the number of flagged records.
* The 60-minute and 120-minute re-entry windows produced the same number of records with nonzero SHRR in the reported analysis.
* Several activity and transition features were strongly correlated, indicating redundancy in the feature representation.

### Synthetic Benchmark

A separate synthetic benchmark was used to check model behaviour on constructed activity patterns. The extended feature model achieved perfect scores on that synthetic test, while the activity baseline performed less strongly.

These synthetic labels were constructed from the same patterns that the features were designed to measure. Therefore, the result is an implementation check, **not evidence of perfect real-world anomaly detection**.

## Technologies Used

* **Python** — data processing and feature engineering
* **pandas** — event cleaning, aggregation, and analysis
* **NumPy** — numerical computation
* **scikit-learn** — Isolation Forest and evaluation
* **Matplotlib** — experimental visualizations
* **GH Archive** — public GitHub event data
* **LaTeX / IEEE format** — research paper preparation
* **Git and GitHub** — version control and project hosting

## Repository Structure

The repository is organized around the implementation, experiment outputs, and research paper.

```text
developer-workflow-fragmentation/
├── data/
│   ├── raw/                 # Local raw data; excluded from Git
│   └── processed/           # Local processed data; excluded from Git
├── src/                     # Source code, if organized separately
├── results/                 # Selected experiment outputs
├── paper/                   # IEEE paper and figures, if included
├── README.md
├── requirements.txt         # Python dependencies, if provided
└── .gitignore
```

The exact directory layout may differ depending on the files included in the repository. Refer to the actual source files for the implementation.

## Reproducibility

To reproduce the analysis:

1. Clone this repository.
2. Set up a Python environment.
3. Install the dependencies listed in `requirements.txt`, if available.
4. Obtain the required public event archives.
5. Run the preprocessing and feature-engineering scripts using the documented arguments.
6. Execute the model comparison and sensitivity analyses.
7. Compare the resulting outputs with the reported experimental results.

The results depend on the sampled event files, preprocessing rules, feature definitions, and model configuration. Confirm these details before attempting to reproduce the reported numbers.

## Limitations

This study has several limitations:

* The analysis uses a sparse sample of public GitHub events from a single date.
* Public events do not capture all development activities.
* Automated accounts remain in the dataset.
* Some activity and transition features are highly correlated.
* The real-data anomaly flags have no independently verified ground-truth labels.
* Synthetic benchmark results do not establish real-world model performance.
* Repository switching alone cannot establish distraction, productivity, stress, burnout, or cognitive load.

These limitations mean that the results should be interpreted as exploratory evidence about model behaviour on the observed data.

## Future Work

Potential extensions include:

* Collecting a longer and more continuous observation window.
* Establishing a documented bot-filtering strategy.
* Testing feature ablations to measure the contribution of individual features.
* Comparing additional unsupervised anomaly-detection methods.
* Evaluating flagged records using independent labels or a transparent human-review protocol.
* Improving reproducibility by recording exact dependencies, parameters, and data provenance.

## Research Paper

The IEEE-format paper and its figures can be included in a `paper/` directory. If the LaTeX source is provided, compile it using Overleaf or a local LaTeX distribution.

## References

1. GH Archive, [The public GitHub timeline](https://www.gharchive.org/).
2. F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation Forest," *Proceedings of the 8th IEEE International Conference on Data Mining*, 2008, pp. 413–422. https://doi.org/10.1109/ICDM.2008.17.
3. E. Kalliamvakou et al., "The promises and perils of mining GitHub," *Proceedings of the 11th Working Conference on Mining Software Repositories*, 2014, pp. 92–101. https://doi.org/10.1145/2597073.2597074.
4. M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, "LOF: Identifying density-based local outliers," *Proceedings of the ACM SIGMOD International Conference on Management of Data*, 2000, pp. 93–104. https://doi.org/10.1145/342009.335388.

---

**Author:** K. Sruthi
**Institution:** VIT-AP University, Amaravati, India

---

*This repository is an academic research project. The findings describe exploratory anomaly detection on sampled public GitHub activity and should not be interpreted as validated measurements of individual developer behaviour.*
