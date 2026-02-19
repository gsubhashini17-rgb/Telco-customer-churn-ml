# Customer Churn Prediction using Machine Learning

## Project Overview

Customer churn is a critical problem for subscription-based businesses.
This project aims to build a predictive model to identify customers who are likely to churn, enabling targeted retention strategies.

The solution follows a complete end-to-end Machine Learning pipeline including EDA, preprocessing, imbalance handling, model tuning, and evaluation.

## Business Objective

* Identify high-risk customers before churn
* Improve retention campaigns
* Reduce revenue loss
* Support data-driven decision making

## Dataset

* **Dataset:** Telco Customer Churn Dataset
* **Records:** ~7,000 customers
* **Target Variable:** `Churn` (Yes/No → 1/0)
* **Features Include:**

  * Tenure
  * Contract type
  * Monthly charges
  * Internet service
  * Payment method
  * Total charges

## Exploratory Data Analysis (EDA)

Key findings:

* Dataset was imbalanced (~26% churn rate)
* Customers with month-to-month contracts had higher churn
* Higher monthly charges correlated with increased churn risk
* Tenure showed strong inverse relationship with churn

Visualizations included:

* Count plots
* Box plots
* Correlation heatmap

**Data Preprocessing**

* Converted `TotalCharges` to numeric
* Removed missing values
* Dropped non-informative ID column
* One-hot encoded categorical variables
* Stratified train-test split (80/20)
* Handled class imbalance using **SMOTE**
* Applied **StandardScaler** for feature normalization

To prevent data leakage during cross-validation, preprocessing steps were implemented inside an imblearn Pipeline.

**Models Implemented**

1️⃣ Logistic Regression (Baseline)

* ROC-AUC: ~0.81
* Evaluated using confusion matrix, precision, recall, F1-score

2️⃣ Random Forest (Baseline)

* Compared performance against Logistic Regression

3️⃣ Hyperparameter Tuning (GridSearchCV)

* 5-fold cross-validation
* Scoring metric: ROC-AUC
* Tuned parameter: Regularization strength (`C`)

Best Parameters:
C = 1
penalty = l2
solver = lbfgs

Final Model Performance

| Metric                   | Value    |
| ------------------------ | -------- |
| Cross-Validation ROC-AUC | **0.83** |
| Test ROC-AUC             | **0.82** |
| Accuracy                 | ~0.77    |
| Recall (Churn Class)     | ~0.60    |

The close CV and test ROC-AUC scores indicate good generalization and minimal overfitting.

Evaluation Metrics Used

* Confusion Matrix
* Precision
* Recall
* F1-Score
* ROC Curve
* ROC-AUC

ROC-AUC was prioritized due to class imbalance.

Key Technical Concepts Applied

* Stratified Train-Test Split
* SMOTE for imbalance handling
* StandardScaler for normalization
* Pipeline integration to avoid data leakage
* Hyperparameter tuning with GridSearchCV
* Bias-Variance tradeoff understanding
* Threshold-based classification


Business Impact

The model can be used to:

* Identify high-risk customers
* Trigger proactive retention offers
* Reduce churn-related revenue loss
* Support customer segmentation strategies

Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)

🚀 Future Improvements

* Threshold tuning for recall optimization
* XGBoost model comparison
* Cost-sensitive learning
* Deployment using Streamlit
* Model monitoring and drift detection

👤 Author

Subhashini G
AI/ML Practitioner | Data Science Enthusiast



Tell me your next step.
