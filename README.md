# Tribology Data Analysis and Machine Learning Pipeline

This repository contains a Python-based research framework designed for processing, analyzing, and modeling tribological measurement data. The system focuses on predicting the Coefficient of Friction (COF) and Friction Absolute Integral (FAI) using various machine learning architectures, complemented by Design of Experiments (DoE) suggestions and SHAP-based interpretability.

## System Overview

The pipeline automates the transition from raw Excel measurement files to a comprehensive research report. It includes specialized modules for physical feature engineering, such as calculating Hertzian contact stress and generating interaction terms between load, temperature, and concentration.

### Core Components

* Data Processing & Filtering: Automated loading of multi-sheet Excel files with rolling mean noise reduction and percentile-based outlier removal.
* Feature Engineering: Implementation of custom transformers for VIF-based (Variance Inflation Factor) multicollinearity filtering and interaction feature generation.
* Model Suite: A diverse set of regressors including XGBoost, LightGBM, CatBoost, Random Forest, and Multi-Layer Perceptrons (MLP), often wrapped in log-transform pipelines for target stabilization.
* Optimization & DoE: Bayesian hyperparameter tuning via Optuna and an uncertainty-driven DoE module that suggests new measurement points based on model variance and spatial sparsity.
* Reporting: Generation of interactive 3D distribution plots and detailed HTML reports summarizing model performance, Pareto fronts, and physical trend analyses.

## Project Structure

* `main.py`: The central execution script that orchestrates data loading, model training, and result generation.
* `config.py`: Centralized configuration for physical constants (e.g., E = 210000 MPa), file paths, academic plot styles, and model hyperparameters.
* `transformers.py`: Custom Scikit-learn compatible classes for data scaling, interaction term generation, and VIF selection.
* `utils.py`: Helper functions for time formatting, Pareto front calculation, and learning curve visualization.

## Technical Details

The framework utilizes a TransformedTargetRegressor approach to handle the non-linear nature of friction data, applying a natural logarithm transformation to targets before training. Evaluation is performed using GroupKFold cross-validation to ensure the models generalize across different experimental files rather than just individual data points.

For physical accuracy, the contact mechanics are modeled using the Hertzian stress formula:
P_max = (3 * F) / (2 * pi * a^2)
where 'a' represents the contact radius derived from the ball geometry and material elastic properties defined in the configuration.

## Outputs

Upon execution, the system populates a Results directory with:
1. HTML Report: A standalone summary containing all metrics and descriptions.
2. Visualization Suite: SHAP impact plots, 3D data coverage maps, and COF heatmaps.
3. Trained Model: The best performing pipeline saved as a joblib pickle for future inference.
4. Excel Tables: Raw numerical data for further statistical processing.
