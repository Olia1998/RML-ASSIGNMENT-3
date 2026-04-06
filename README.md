# RML-ASSIGNMENT-3
# Fairness Analysis of Predictive Risk Models

## Overview

This project evaluates fairness and potential bias in two predictive machine learning models: **Logistic Regression** and **Gradient-Boosted Trees (GBT)**. The analysis examines whether model predictions differ across demographic groups, particularly **race** and **sex**.

Several fairness metrics are computed, including **Adverse Impact Ratio (AIR)**, **Marginal Effect (ME)**, **Standardized Mean Difference (SMD)**, and **error rate disparities** such as **False Positive Rate (FPR)** and **False Negative Rate (FNR)**. The results help identify whether certain demographic groups are disproportionately affected by model predictions.

The analysis also includes **intersectional fairness evaluation** (race × gender) and statistical testing using **two-proportion z-tests** to determine whether observed disparities are statistically significant.

---

## Purpose of the Analysis

The purpose of this analysis is to:

* Evaluate **algorithmic fairness** in predictive risk models.
* Identify disparities in predictions across **race and sex groups**.
* Examine **intersectional subgroups** (race × gender) to detect compounded disparities.
* Test whether differences in model error rates across groups are **statistically significant**.
* Provide visualizations to clearly communicate disparities in model performance.

This helps ensure that predictive models are used responsibly and that any potential bias is identified and documented.

---

## Python Libraries Used

The following Python libraries were used in this analysis:

* **pandas** – data manipulation and analysis
* **numpy** – numerical computations
* **matplotlib** – data visualization
* **seaborn** – advanced statistical plotting
* **statsmodels** – statistical testing (two-proportion z-test)
* **scikit-learn** – machine learning models and predictions
* **solas-ai** – fairness metric computation (AIR, ME, SMD)

These libraries support data processing, model evaluation, fairness analysis, and visualization.

---

## Analysis Components

### 1. Standardized Mean Difference (SMD)

SMD is calculated to measure differences in predicted probabilities between **female and male groups**. This metric helps determine whether predicted risk scores differ significantly by gender.

### 2. Intersectional Analysis

An intersectional fairness analysis evaluates combined demographic groups (race × gender). The **Adverse Impact Ratio (AIR)** is calculated for each subgroup relative to the reference group **Caucasian / Male**. The subgroup with the lowest AIR is identified as the most disadvantaged.

### 3. Error Rate Disparities

The analysis computes **False Positive Rate (FPR)** and **False Negative Rate (FNR)** for each racial group. These metrics assess whether certain groups are more likely to be incorrectly classified by the model.

### 4. Statistical Significance Testing

A **two-proportion z-test** is used to determine whether differences in FPR and FNR between demographic groups and the reference group are statistically significant.

### 5. Visualization

Grouped bar charts are generated to visualize FPR and FNR disparities across racial groups, with **Caucasian as the reference group**.

---

## Instructions for Reproducing the Results

1. **Install Required Libraries**

   Install the required packages if they are not already available:

   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
   ```

2. **Prepare the Dataset**

   Ensure the dataset includes the following fields:

   * `race_factor`
   * `gender_factor`
   * model predictions
   * predicted probabilities (`predict_proba`)
   * confusion matrix values (TP, FP, TN, FN)

3. **Run Model Predictions**

   Generate predicted probabilities using both models:

   ```python
   lr_pipeline.predict_proba(X_test)
   gbt_pipeline.predict_proba(X_test)
   ```

4. **Compute Fairness Metrics**

   Run the functions provided in the notebook to calculate:

   * AIR
   * SMD
   * FPR and FNR disparities

5. **Run Statistical Tests**

   Execute the two-proportion z-test function to calculate **z-scores and p-values** for each group comparison.

6. **Generate Visualizations**

   Run the plotting function:

   ```python
   plot_fpr_fnr_disparity(lr_disparity_analysis, "Logistic Regression")
   plot_fpr_fnr_disparity(gbt_disparity_analysis, "Gradient-Boosted Model")
   ```

   This will produce grouped bar charts comparing FPR and FNR across racial groups.

---

## Output

The analysis produces:

* Tables of fairness metrics (AIR, SMD)
* Intersectional subgroup analysis
* FPR/FNR disparity tables with statistical significance
* Publication-quality visualizations

These outputs help identify potential **algorithmic bias** and support transparency in model evaluation.

