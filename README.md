# ML_Project_hackathon
# 🎯 High-Value Customer Intelligence (HVCI) Pipeline

### An end-to-end classification pipeline for identifying high-value customers, built for a hackathon.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-orange)
![scikit--learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikit-learn&logoColor=white)

## 📖 About

This project predicts which customers are likely to make a high-value purchase, using a dataset of 5,000 customers with 16 behavioral and demographic features. The goal was to build the most reliable classifier possible under hackathon time constraints — going beyond a single model into a tuned ensemble.

## 🏆 Built For

Developed for the **"Building AI with LLM's" Workshop 2026** — organized by the IEEE Photonics & ComSoc Joint Chapter in association with the Department of Electronics and Telecommunication Engineering, Siddaganga Institute of Technology.

🥈 Secured the **Runner's title**, recognized for problem-solving and teamwork under time constraints.

*Dataset provided by the workshop organizers.*
## 🧠 Approach

**1. Feature Engineering**
- RFM-style scoring (recency, frequency, monetary value) combining purchase history and account age
- Engineered features: `total_revenue`, `value_per_month`, `purchase_velocity`, `advocacy_score` (reviews × rating), `intent_clarity` (email engagement vs. bounce rate)
- Automated customer segmentation via **K-Means clustering** on age, revenue, and engagement score

**2. Preprocessing**
- Median imputation for numeric features, constant imputation for categorical
- Standard scaling + one-hot encoding via a `ColumnTransformer`
- **SMOTE** oversampling to address class imbalance within the training pipeline

**3. Model Training & Tuning**
- Three base models tuned via `RandomizedSearchCV` with 5-fold stratified cross-validation:
  - XGBoost
  - Random Forest
  - Logistic Regression

**4. Ensembling**
- **Soft Voting Classifier** combining the top 2 performing tuned models
- **Stacking Classifier** combining all three base models with a logistic regression meta-learner

**5. Threshold Optimization**
- Rather than using the default 0.5 cutoff, the classification threshold was tuned per model to maximize F1-score using the precision-recall curve

## 📊 Results

| Model | F1-Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **XGBoost (tuned)** | **0.817** | 0.795 | 0.840 | **0.903** |
| Stacking Ensemble | 0.815 | 0.829 | 0.802 | 0.903 |
| Soft Voting Ensemble | 0.813 | 0.733 | 0.912 | 0.900 |
| Random Forest (tuned) | 0.812 | 0.781 | 0.844 | 0.893 |
| Logistic Regression (tuned) | 0.809 | 0.758 | 0.866 | 0.887 |

The tuned XGBoost model performed best on F1 and ROC-AUC individually, though the stacking ensemble offered the best precision — useful if the business cost of false positives is high.
 
## 🛠️ Tech Stack

- **Data & ML:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Visualization:** matplotlib, seaborn
- **Environment:** Google Colab

## 📁 Files

- `hvci_customer_intelligence_model.ipynb` — full pipeline: EDA, feature engineering, model tuning, ensembling, evaluation
- `requirements.txt` — dependencies

## 👤 Author

**Nishanth H**
- GitHub: [@Nishanth-H05](https://github.com/Nishanth-H05)
- LinkedIn: [Nishanth H](https://www.linkedin.com/in/nishanthh-141540386)
