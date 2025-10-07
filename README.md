# BIMEC-code
code for Echo Chamber-Aware Budgeted Influence Maximization on Social Media (BIMEC)

# Overview
```
BIMEC/
  BIMEC_code/
    Datasets/
      data_preprocess/
        askubuntu.ipynb
        youtube.ipynb
    ECBIM/
      UserCost_Calculate/
        calculate_cost.py
      big_element.py
      budget_allocation.py
      ccar_calculate.py
      data_loader.py
      influence_maximization.py
      main.py
      monte_carlo.py
    EdgeProcess/
      GCN_MLP.py
      load_data.py
      visulization_comparison.ipynb
LICENSE
README.md
```
1. askubuntu.ipynb; youtube.ipynb: used to filter the original data
2. calculate_cost.py: used to calculate user cost under different `$\lambda$`
3. big_element.py: enumerate large elements to test the performance
4. budget_allocation.py: allocate budget to each community
5. ccar_calculate.py: calculate and return the CCAR result
6. influence_maximization.py: performs influence maximization using CELF with RR-set–based estimation
7. main.py: serves as the entry point of the BIMEC framework.
8. GCN_MLP.py: provides a method for edge probability construction
9. visulization_comparison.ipynb: visualization of different edge construction methods

# Usage
To run this program, please follow the steps below:
first `cd BIMEC/BIMEC_code`

**1. Obtain the dataset**
Please download the datasets from the SNAP website and place them in the Datasets/ directory.

**2. Preprocess the data**
Run the preprocessing scripts provided in the data_preprocess/ folder to filter the raw data for subsequent stages:
`python Datasets/data_preprocess/prepare_dataset.py`

**3. Calculate user cost**
Generate user-level cost:
`python ECBIM/UserCost_Calculate/calculate_cost.py`

**4. Construct edge probabilities**
Train the GCN-MLP model to estimate edge weights:
`python  EdgeProcess/GCN_MLP.py`

**5. Run the BIMEC framework**
`python ECBIM/main.py`

