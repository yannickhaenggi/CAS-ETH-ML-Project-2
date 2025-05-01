# CAS-ETH-ML-Project-2
# Insurance Claim Prediction  
*Introduction to Machine Learning in Finance and Insurance – ETH Zurich, Spring 2024*  
**First discussion:** Apr 12  **Deadline:** May 17

The goal of this project is to implement and compare different models for insurance frequency claim prediction on real-life data from the French motor third party liability dataset (`freMTPL2freq.csv`), which contains **678,007 car insurance policies**.

## Dataset Overview

| Feature Name  | Description                                   | Type / Range                                |
|---------------|-----------------------------------------------|---------------------------------------------|
| VehPower      | Car power                                     | Discrete, {4, 5, ..., 15}                   |
| VehAge        | Car age in years                              | Discrete, {0, 1, ..., 100}                  |
| DrivAge       | Driver’s age in years                         | Discrete, {18, ..., 100}                    |
| BonusMalus    | Bonus-malus level                             | Discrete, starts at 100                     |
| VehBrand      | Car brand                                     | Categorical, {'B1', ..., 'B14'}             |
| VehGas        | Fuel type                                     | Categorical, {diesel, regular}              |
| Density       | Population density (inhab/km²)                | Discrete, from 1 to 27,000                  |
| Region        | Region of residence                           | Categorical, {R11, ..., R94}                |

| Label Name  | Description                                   | Type / Range                        |
|-------------|-----------------------------------------------|-------------------------------------|
| Exposure    | Policy duration (years)                       | Continuous, [0, 1]                  |
| ClaimNb     | Number of insurance claims                    | Discrete, {0, 1, ..., 5}            |

---

## 1. Data Exploration

- Plot features and claim frequency:  
  \[
  y_i = \frac{\text{ClaimNb}_i}{\text{Exposure}_i}
  \]
- Use pairplots and other exploratory techniques.
- Include relevant analysis and observations.

---

## 2. Poisson GLM

We fit a **Poisson Generalized Linear Model (GLM)** commonly used in insurance.

Let:
- \( y_i = \frac{\text{ClaimNb}_i}{\text{Exposure}_i} \)
- \( \lambda_i = \exp\left(\sum_{j=1}^{d} \theta_j x_{ij} + \theta_0\right) \)

Assume:
- \( y_i \cdot \text{Exposure}_i \sim \text{Poisson}(\lambda_i \cdot \text{Exposure}_i) \)

### Loss Function (Exposure-weighted Poisson Deviance):

\[
L(D, \hat{\theta}) = \frac{1}{\sum_{i=1}^{m} \text{Exposure}_i} \sum_{i=1}^{m} \text{Exposure}_i \cdot \ell(\hat{\lambda}_i, y_i)
\]

with:
\[
\ell(\hat{\lambda}, y) = 2\left(\hat{\lambda} - y - y \log(\hat{\lambda}) + y \log(y)\right)
\]

### (a) Feature Pre-processing

- VehPower → log(VehPower)  
- VehAge → Categorical: [0,6), [6,13), [13,∞)  
- DrivAge → log(DrivAge)  
- BonusMalus → log(BonusMalus)  
- Density → log(Density)

### (b) Train/Test Split and Model Training

- Use 90% train / 10% test split
- Use `sklearn.linear_model.PoissonRegressor` with `alpha=0`
- Fit with `Exposure` as sample weight
- Report:  
  - Weighted MAE  
  - Weighted MSE  
  - Loss \(L\) (on train/test)

*Note: Standardize continuous/discrete features, apply one-hot encoding to categorical features.*

### (c) Add Engineered Features

Include:
- `DrivAge²`
- `BonusMalus × DrivAge`
- `BonusMalus × DrivAge²`

Evaluate improvement in GLM performance.

---

## 3. Poisson Feedforward Neural Network Model

Still under the Poisson assumption, now use a **neural network** for modeling.

### (a) Model Implementation

- Architecture suggestion:
  - 2 hidden layers, 20 neurons each
  - ReLU activation in hidden layers
  - Exponential activation in output
- Training:
  - 100 epochs
  - Batch size: 10,000
  - Learning rate: 0.01

*(Bonus)*: Use Keras Tuner for tuning:  
https://www.tensorflow.org/tutorials/keras/keras_tuner

### (b) Model Training

- Use Keras and compile with `keras.losses.Poisson`
- Fit with `Exposure` as sample weight
- Report:
  - Weighted MAE  
  - Weighted MSE  
  - Loss \(L\) (train/test)
- Compare with GLM

*(Bonus)*: Try L2 regularization. Tune using grid search.
