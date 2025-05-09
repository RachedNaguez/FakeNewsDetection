# Fake News Detection

A machine learning-based system for detecting fake news articles using natural language processing and classification algorithms.

## Overview

This project aims to combat the spread of misinformation by providing an automated tool that can analyze news articles and determine their credibility. The system uses natural language processing techniques to extract features from text and employs machine learning models to classify content as reliable or potentially false.

This is an end-to-end MLOps project utilizing ZenML, MLflow, and Streamlit to create a robust machine learning pipeline with production-ready capabilities.

## Features

- Text preprocessing and feature extraction
- Machine learning classification models
- Performance evaluation metrics
- User-friendly interface for article verification
- Complete MLOps pipeline with experiment tracking
- Interactive Streamlit dashboard for model interaction

## Technologies Used

- Python
- Scikit-learn
- NLTK/spaCy for NLP
- Pandas for data manipulation
- ZenML for pipeline orchestration
- MLflow for experiment tracking and model registry
- Streamlit for interactive web interface

## ML Pipeline

This project implements a comprehensive machine learning pipeline with the following steps:

1. **Exploratory Data Analysis (EDA)**: Thorough analysis of the dataset to understand patterns, distributions, and relationships within the data.

2. **Feature Engineering**: Transformation of raw text data into meaningful features using NLP techniques including tokenization, vectorization, and sentiment analysis.

3. **Model Implementation and Validation**: Training various classification models and validating their performance using cross-validation and appropriate metrics.

4. **MLOps Integration**: Production-ready implementation using:
   - ZenML for pipeline orchestration and reproducibility
   - MLflow for experiment tracking, model versioning, and deployment
   - Streamlit for creating an interactive user interface

5. **Scalable and Readable Code**: Implementation of design patterns and best practices for clean, efficient, and maintainable code.

## Installation

## Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```

Starting with ZenML 0.20.0, ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally, but first you must install the optional dependencies for the ZenML server:

```bash
pip install zenml["server"]
zenml up
```

If you are running the run_deployment.py script, you will also need to install some integrations using ZenML:

```bash
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

## Diving into the code

You can run two pipelines as follows:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```

- Demo Streamlit App:

```bash
streamlit run stream.py
```
