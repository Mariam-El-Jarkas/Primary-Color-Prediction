# Primary Color Prediction

## Overview
This project investigates the use of machine learning techniques to analyze a single-color image and predict its composition in terms of primary colors (Red, Blue, Yellow). The system addresses both categorical and continuous prediction tasks by combining classification and regression approaches.

The objective is to compare multiple models, evaluate their performance under varying noise conditions, and assess their suitability for color composition analysis.

---

## Problem Statement
Given a single solid color represented by RGB values, the project aims to:

1. **Classify** which primary colors were used to create the input color.
2. **Estimate** the proportional contribution of each primary color.

---

## Approach

### Input Representation
- Each input color is represented using RGB values.
- Each sample corresponds to a single-color image.

### Outputs
- **Classification:** One label representing the set of primary colors used.
- **Regression:** Continuous values representing the percentage contribution of Red, Blue, and Yellow.

---

## Models Used

### Classification
The following classification algorithms were implemented and evaluated:
- K-Nearest Neighbors (KNN)
- Naïve Bayes
- Decision Tree
- Random Forest

### Regression
- Linear Regression was used to estimate primary color mixing ratios.

---

## Dataset
- The dataset was synthetically generated to ensure full control over color composition.
- Each sample includes:
  - RGB values
  - A categorical label indicating the primary color combination
  - Ground-truth percentage contributions of each primary color
- Noise was optionally added to RGB values to evaluate model robustness.

---

## Evaluation Metrics

### Classification
- Accuracy
- F1-score
- Confusion matrix
- Performance under varying noise levels

### Regression
- Coefficient of determination (R²)
- Mean Absolute Error (MAE)
- Per-channel performance analysis

---

## Results Summary

### Classification Performance
- Decision Trees achieved the highest accuracy on clean data.
- Random Forests demonstrated a strong balance between accuracy and robustness.
- Naïve Bayes showed lower overall accuracy but was the most resilient to noise.
- KNN performed well on clean data but degraded with increasing noise.

### Regression Performance
- Linear Regression achieved an overall R² of 0.87.
- Feature engineering significantly improved prediction accuracy, particularly for the yellow color component.
- Mean absolute prediction error remained within approximately 4%.

---

## Project Structure
//To be added

---

## Applications
- Educational tools for teaching color theory
- Digital art and design systems
- Color analysis in image processing pipelines

---

## Limitations
- The dataset is synthetically generated and may not fully represent real-world image distributions.
- The analysis is limited to single-color images.

---

## Future Work
- Extension to multi-color images
- Use of alternative color spaces (HSV, LAB)
- Evaluation on real-world image datasets
- Exploration of advanced regression models

---

## Author
Mariam El Jarkas
