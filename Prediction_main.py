# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:22:46 2024

@author: Jamali
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import load
import matplotlib.pyplot as plt

# Load saved models, scaler, and feature selectors
scaler = load("scaler.joblib")
svm_model_anova = load("svm_model_anova.joblib")
svm_model_woa = load("svm_model_woa.joblib")
feature_selector_anova = load("feature_selector_anova.joblib")
feature_selector_woa = load("feature_selector_woa.joblib")

# Load new data and preprocess
def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)  # Read from Excel
    X = data.drop(columns=['Wear Condition', 'Folder Name'])
    y = data['Wear Condition']
    X_scaled = scaler.transform(X)
    return X_scaled, y

# Function to select features, predict with each model, and save results to Excel
def predict_and_evaluate(X_scaled, y_true, output_file):
    # ANOVA model predictions
    X_anova = X_scaled[:, feature_selector_anova]
    y_pred_anova = svm_model_anova.predict(X_anova)
    accuracy_anova = accuracy_score(y_true, y_pred_anova)
    conf_matrix_anova = confusion_matrix(y_true, y_pred_anova)

    # WOA model predictions
    X_woa = X_scaled[:, feature_selector_woa]
    y_pred_woa = svm_model_woa.predict(X_woa)
    accuracy_woa = accuracy_score(y_true, y_pred_woa)
    conf_matrix_woa = confusion_matrix(y_true, y_pred_woa)

    # Print accuracy scores
    print(f"Accuracy for ANOVA-selected features: {accuracy_anova * 100:.2f}%")
    print(f"Accuracy for ANOVA+WOA-selected features: {accuracy_woa * 100:.2f}%")

    # Plot predictions vs. actual values
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual Wear Condition', color='black')
    plt.plot(y_pred_anova, label='Predicted Wear Condition - ANOVA', linestyle='--', color='blue')
    plt.plot(y_pred_woa, label='Predicted Wear Condition - ANOVA+WOA', linestyle=':', color='green')
    plt.xlabel("Sample Index")
    plt.ylabel("Wear Condition")
    plt.legend()
    plt.title("Comparison of Actual and Predicted Wear Conditions")
    plt.show()

    # Save results to an Excel file
    with pd.ExcelWriter(output_file) as writer:
        # Save accuracies
        pd.DataFrame({
            "Model": ["ANOVA", "ANOVA+WOA"],
            "Accuracy": [accuracy_anova, accuracy_woa]
        }).to_excel(writer, sheet_name="Accuracy", index=False)

        # Save confusion matrices
        pd.DataFrame(conf_matrix_anova, columns=["Pred 1", "Pred 2", "Pred 3"], index=["Actual 1", "Actual 2", "Actual 3"]).to_excel(writer, sheet_name="Confusion Matrix ANOVA")
        pd.DataFrame(conf_matrix_woa, columns=["Pred 1", "Pred 2", "Pred 3"], index=["Actual 1", "Actual 2", "Actual 3"]).to_excel(writer, sheet_name="Confusion Matrix WOA")

        # Save predictions and actual values for further analysis
        results_df = pd.DataFrame({
            "Actual Wear Condition": y_true.values,
            "Predicted Wear Condition - ANOVA": y_pred_anova,
            "Predicted Wear Condition - ANOVA+WOA": y_pred_woa
        })
        results_df.to_excel(writer, sheet_name="Predictions", index=False)

    print(f"Results saved to {output_file}")

    return accuracy_anova, accuracy_woa, conf_matrix_anova, conf_matrix_woa

# Main function to run predictions, plot results, and save them
def main():
    # Load and preprocess the new data
    file_path = input("Enter the path to the new data Excel file: ")
    X_scaled, y_true = load_and_preprocess_data(file_path)
    
    # Specify the output file path for results
    output_file = input("Enter the path to save the results Excel file (e.g., results.xlsx): ")
    
    # Predict and evaluate with both models, and save results to Excel
    accuracy_anova, accuracy_woa, conf_matrix_anova, conf_matrix_woa = predict_and_evaluate(X_scaled, y_true, output_file)
    
    # Print confusion matrices for reference
    print("Confusion Matrix for ANOVA-selected features:")
    print(conf_matrix_anova)
    print("Confusion Matrix for ANOVA+WOA-selected features:")
    print(conf_matrix_woa)

if __name__ == "__main__":
    main()
