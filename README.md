# Credit Card Fraud Detector

This repository contains a single Python script that demonstrates how to build and evaluate a machine learning model for detecting credit card fraud, using the well-known Kaggle credit card fraud dataset.

## Overview

The script:
- Loads the dataset (`creditcard.csv`)
- Splits the data into training and testing sets
- Optionally includes (commented out) steps for scaling and SMOTE oversampling
- Trains an XGBoost classifier with class imbalance handling (`scale_pos_weight`)
- Evaluates performance with a classification report and confusion matrix

## File

- `Credit_card_frad_model.py` – All code is in this file

## How to Use

1. **Prerequisites:**
    - Python 3.7 or newer
    - Install the following Python packages manually if not present:
        - pandas
        - scikit-learn
        - xgboost

    To install all dependencies:
    ```bash
    pip install pandas scikit-learn xgboost
    ```

2. **Dataset:**
    - Download the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
    - Place the `creditcard.csv` file in the same directory as the script.

3. **Run the script:**
    ```bash
    python Credit_card_frad_model.py
    ```

4. **Output:**
    - The script will print class counts in the training set and a classification report with model evaluation metrics.

## Notes

- Some parts (data scaling and SMOTE for balancing) are included as commented code. You may uncomment these for experimentation with balancing techniques or feature scaling.
- The script uses XGBoost, which handles class imbalance using the calculated `scale_pos_weight` parameter.



## References

- [Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
