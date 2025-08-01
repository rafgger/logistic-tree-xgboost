# Credit Risk Prediction: Logistic Regression vs Tree-Based Models

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

## 📋 Overview

This project presents a comprehensive analysis of credit risk prediction using multiple machine learning approaches. The study compares the performance of **Logistic Regression**, **Random Forest**, and **XGBoost** models for predicting problematic loans using the Kaggle Credit Card Approval Prediction dataset.

## 🎯 Key Findings

- **Best Model**: Random Forest achieved **0.780 AUC** with optimal balance of precision and recall
- **Business Impact**: Model can prevent **40% of bad loan approvals** while maintaining **97% approval rate**
- **Data Quality**: Successfully cleaned and engineered features from 36,457 applications
- **Feature Importance**: Employment history, age, and income ratios are key predictors

## 📊 Results Summary

| Model | AUC Score | Precision | Recall | Recommendation |
|-------|-----------|-----------|--------|----------------|
| **Random Forest** | **0.780** | 22.5% | 39.6% | **Production Ready** |
| XGBoost | 0.775 | 18.9% | 45.3% | High Recall Alternative |
| Decision Tree | 0.724 | 20.1% | 35.2% | Interpretable Option |
| Logistic Regression | 0.581 | 15.8% | 28.7% | Baseline Model |

## 🛠️ Technical Features

### Data Processing
- ✅ **Outlier Detection & Handling**: IQR-based approach with 83% data retention
- ✅ **Feature Engineering**: 13 new predictive features including ratio-based metrics
- ✅ **Missing Value Treatment**: Median imputation for numerical stability
- ✅ **Categorical Encoding**: Label encoding for optimal model performance

### Model Validation
- ✅ **Cross-Validation**: 5-fold stratified validation for robust evaluation
- ✅ **Statistical Tests**: Box-Tidwell linearity test and VIF multicollinearity analysis
- ✅ **Performance Metrics**: Comprehensive evaluation with ROC curves, precision-recall, and business impact analysis

## 📁 Project Structure

```
├── logistic-tree-xgboost.ipynb    # Main analysis notebook
├── README.md                      # This file
└── data/                         # Dataset (downloaded via kagglehub)
```

## 🚀 Quick Start

### Prerequisites
```python
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels kagglehub
```

### Run the Analysis
1. Clone this repository
2. Open `logistic-tree-xgboost.ipynb` in Jupyter
3. Run all cells to reproduce the analysis

### Dataset
The analysis uses the [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) dataset from Kaggle, automatically downloaded via `kagglehub`.

## 📈 Model Performance Details

### Random Forest (Best Model)
- **AUC**: 0.780 - Good discriminative ability
- **Business Impact**: Identifies 40% of problematic loans
- **Efficiency**: 3% rejection rate for 1.7% actual bad rate
- **Feature Importance**: Age, employment stability, income ratios

### Key Performance Metrics
- **Precision**: 22.5% (1 in 4 predicted bad clients is actually bad)
- **Recall**: 39.6% (catches 40% of actual bad clients)  
- **F1-Score**: Balanced performance across metrics
- **Cross-Validation**: Consistent performance across folds

## 🔍 Business Applications

### Risk Management Strategy
1. **Conservative**: Lower threshold → catch more bad clients, higher rejection rate
2. **Balanced**: Current threshold → optimal precision-recall trade-off  
3. **Aggressive**: Higher threshold → approve more applications, accept more risk

### Expected Outcomes
- **40% reduction** in bad loan approvals
- **Maintained approval rates** (~97% of applications)
- **Improved portfolio quality** through data-driven decisions
- **Scalable risk assessment** for growing application volumes

## 📚 Methodology

### Statistical Validation
- **Box-Tidwell Test**: Verified logistic regression linearity assumptions
- **VIF Analysis**: Confirmed low multicollinearity (all VIF < 5)
- **Feature Transformations**: Applied where linearity was violated

### Model Development Process
1. **Data Exploration**: Comprehensive EDA with correlation analysis
2. **Feature Engineering**: Created predictive ratio and categorical features
3. **Model Training**: Trained multiple algorithms with proper validation
4. **Performance Evaluation**: Business-focused metric interpretation
5. **Production Readiness**: Final model selection and deployment recommendations

## 💡 Key Insights

### Risk Factors Identified
- **Age**: Older applicants tend to be lower risk
- **Employment Stability**: Long-term employment reduces default probability
- **Income-to-Age Ratio**: Financial maturity indicator
- **Family Structure**: Affects financial stability assessment

### Business Intelligence
- **Target Rate**: 1.66% overall default rate in the dataset
- **Geographic Patterns**: Regional risk variations identified
- **Demographic Trends**: Age and employment status strongly predictive

## 🎯 Future Enhancements

- [ ] **Advanced Modeling**: Deep learning approaches for complex patterns
- [ ] **Real-time Scoring**: API development for production deployment  
- [ ] **Ensemble Methods**: Combine multiple models for improved performance
- [ ] **Explainable AI**: SHAP values for individual prediction explanations
- [ ] **Monitoring Dashboard**: Real-time model performance tracking

## 📊 Data Sources & References

### Literature
- Hastie, T., Tibshirani, R., & Friedman, J. (2010). *The Elements of Statistical Learning* (2nd ed.)
- Logistic Regression Classifier Tutorial, *Kaggle*. Banerjee, P., 2019
- Credit risk assessment methodologies and best practices

### Dataset Information
- **Source**: Kaggle Credit Card Approval Prediction
- **Size**: 30,322 records after cleaning
- **Features**: 20 columns (19 features + 1 target)
- **Target Distribution**: 98.3% good clients, 1.7% bad clients

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions about this analysis or collaboration opportunities, please open an issue in this repository.

---

⭐ **Star this repository if you find it helpful!**
