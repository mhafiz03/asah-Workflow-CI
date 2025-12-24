name: Build Model

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python 3.9
      uses: actions/setup-python@v4
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pandas scikit-learn mlflow

    - name: Run MLProject
      env:
        MLFLOW_TRACKING_URI: "file://${{ github.workspace }}/mlruns"
      run: |
        mlflow run Workflow-CI/MLProject --env-manager=local
