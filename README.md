# 🏦 Lending Club Loan Defaulters Prediction

A machine learning project for predicting loan defaults using LendingClub loan data. This project implements exploratory data analysis (EDA) and multiple ML models to identify risky loan applicants and minimize credit loss.

## 📋 Table of Contents
- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

LendingClub is a US peer-to-peer lending platform where borrowers can access lower interest rate loans. This project aims to identify patterns that indicate if a person is likely to default on a loan, enabling better risk assessment and decision-making.

## 💼 Business Problem

When approving loan applications, the company faces two types of risks:
1. **Missed Opportunity**: Rejecting applicants who would have repaid (loss of business)
2. **Credit Loss**: Approving applicants who default (financial loss)

The goal is to identify "risky" loan applicants using data-driven approaches to reduce credit loss while maximizing business opportunities.

### Loan Status Categories
- **Fully Paid**: Borrower has fully repaid the loan
- **Current**: Borrower is actively paying instalments
- **Charged Off**: Borrower has defaulted on the loan (target for prediction)

## 📊 Dataset

The dataset contains information about past loan applicants from LendingClub, including:
- Loan details (amount, term, interest rate, grade)
- Borrower information (employment, income, home ownership)
- Credit history (DTI, credit lines, public records)
- Geographic data (address, state, zip code)

**Dataset file**: `lending_club_loan_two.csv`

**Download Dataset**: [LendingClub Dataset on Kaggle](https://www.kaggle.com/datasets/jeandedieunyandwi/lending-club-dataset)

Key features include:
- `loan_amnt`: Loan amount
- `term`: Loan term (36 or 60 months)
- `int_rate`: Interest rate
- `grade` & `sub_grade`: LendingClub assigned loan grades
- `annual_inc`: Borrower's annual income
- `loan_status`: Current loan status (target variable)
- And many more...

## 🔧 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup

1. Clone or download this repository
```bash
cd MLproj
```

2. Install required packages
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Ensure the dataset `lending_club_loan_two.csv` is in the project directory

2. Open and run the Jupyter notebook:
```bash
jupyter notebook lending-club-loan-defaulters-prediction.ipynb
```

3. Follow the notebook cells sequentially to:
   - Load and explore the data
   - Preprocess and clean the data
   - Handle missing values and outliers
   - Engineer features
   - Train and evaluate models
   - Compare model performance

## 🤖 Models

This project implements and compares three machine learning models:

### 1. Artificial Neural Networks (ANNs)
- Multi-layer perceptron classifier
- Hidden layers: (150, 150, 150)
- Early stopping with validation

### 2. XGBoost Classifier
- Gradient boosting ensemble method
- Optimized for classification tasks
- High performance on tabular data

### 3. Random Forest Classifier
- Ensemble of decision trees
- Robust to overfitting
- Feature importance analysis

## 📈 Results

Models are evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: True/False positives and negatives
- **Classification Report**: Precision, recall, and F1-score

Model comparison is visualized using interactive plots with hvplot.

## 📁 Project Structure

```
MLproj/
│
├── lending_club_loan_two.csv              # Dataset
├── lending-club-loan-defaulters-prediction.ipynb  # Main notebook
├── requirements.txt                        # Python dependencies
├── README.md                              # This file
├── instructions.txt                       # Project instructions
├── onepager.pdf                          # Project summary
└── paper.txt                             # Additional documentation
```

## 🔄 Data Processing Pipeline

1. **Data Loading**: Import CSV dataset
2. **Missing Value Handling**: 
   - Drop high-cardinality features (emp_title)
   - Impute mort_acc using total_acc correlation
   - Remove rows with minimal missing data (<0.5%)
3. **Feature Engineering**:
   - Convert categorical to numerical (term)
   - Create dummy variables for categorical features
   - Extract zip codes from addresses
   - Convert date features to year
4. **Outlier Removal**: Filter extreme values in key features
5. **Normalization**: MinMaxScaler for feature scaling
6. **Train-Test Split**: 67-33 split for model validation

## 📝 Key Findings

- Employment length shows minimal correlation with default rates
- Loan sub_grade is a strong predictor (grade is redundant)
- Home ownership, verification status, and purpose are significant factors
- Geographic location (zip code) contributes to prediction accuracy

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements.

## 📄 License

This project is for educational purposes.

## 👥 Authors

Academic Machine Learning Project

## 🙏 Acknowledgments

- LendingClub for providing the dataset
- scikit-learn, XGBoost, and related libraries
