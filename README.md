
# Financial Distress Prediction Project

## Introduction

This project aims to predict the financial distress of companies using a dataset containing financial metrics and performance indicators. The goal is to develop a robust machine learning model to anticipate financial crises and aid decision-making in the financial sector.

## Dataset and Data Exploration

The dataset includes:
- **Financial variables**: financial ratios, revenues, liabilities.
- **Target**: a binary label indicating whether a company is in financial distress.

### Key EDA Insights:
- **Missing values**: [Brief description if applicable].
- **Variable distribution**: [Summary of insights, e.g., the target is imbalanced at 80/20].
- **Feature importance**: Variables like *X* and *Y* were found to be most significant.

Detailed analyses and visualizations are available in the EDA notebook: [Link to notebook].

## Model Training

### Models Used:
1. Logistic Regression.
2. Decision Tree.
3. Random Forest.
4. XGBoost.

### Hyperparameter Tuning:
- GridSearchCV was used to optimize hyperparameters.
- The best-performing model was [model], achieving an accuracy of [score].

Training scripts are available in `train_model.py`.

## Deployment

The model is deployed using Flask, providing a REST API to make predictions from input data.

### Testing the Service:
1. Clone this repository.
2. Install dependencies (see below).
3. Run the Flask service:
   ```bash
   python app.py
   ```
4. Send a POST request to the `/predict` endpoint with a JSON payload containing company features.

## Dependencies and Environment Setup

- Use `pipenv` to manage the environment:
  ```bash
  pipenv install
  pipenv shell
  ```

- Key files:
  - `Pipfile`: Dependency list.
  - `Dockerfile`: For containerization.

## Containerization

A Docker image has been built to simplify deployment. Steps to build and run:
```bash
docker build -t financial-distress-predictor .
docker run -p 5000:5000 financial-distress-predictor
```

## Cloud Deployment

The service is deployed on AWS Elastic Beanstalk. To test:
1. Access the URL: [Link to the deployed application].
2. Send requests using an HTTP client (Postman or curl).

## Reproducibility

To reproduce this project:
1. Download the dataset from [source or link].
2. Run the `eda.ipynb` notebook for data exploration.
3. Train the model using:
   ```bash
   python train_model.py
   ```
4. Follow the deployment instructions.

---
