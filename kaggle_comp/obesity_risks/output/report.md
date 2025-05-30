# 📝 Multi-Class Prediction of Obesity Risk: Tackling Health Through Machine Learning  

## 🚀 Overview  
This competition, **Playground Series - Season 4, Episode 2**, challenges participants to predict obesity risk in individuals using machine learning. The dataset includes factors like age, physical activity, and dietary habits. The aim is multi-class classification of the target variable `NObeyesdad`, which represents five distinct weight categories ranging from underweight to obesity. With cardiovascular disease being a major global concern, this task holds practical significance in public health.  

## 🔧 Data Preprocessing  
Preparing the dataset is a critical step before feeding it to the model. Here's what we did:  
- **Imputation of Missing Values:** Addressed any gaps in the data by:  
  - For numerical features, replacing missing values with the column mean. Example: A missing age value is substituted with the average age of the dataset.  
  - For categorical features, filling missing entries with the most frequent category.  
- **One-Hot Encoding:** Converted categorical variables (e.g., gender) into numerical representations for compatibility with machine learning algorithms. Example: Female → [1, 0], Male → [0, 1].  
- **Normalization (Scaling):** Ensured all numerical features were on a similar scale (e.g., 0 to 1). This prevents variables with large ranges (e.g., income) from dominating the model training.  

Here’s what the preprocessing looked like in code:  
```python
# Impute numerical features
numeric_imputer = SimpleImputer(strategy='mean')
features[numeric_cols] = numeric_imputer.fit_transform(features[numeric_cols])

# Impute categorical features
categorical_imputer = SimpleImputer(strategy='most_frequent')
features[categorical_cols] = categorical_imputer.fit_transform(features[categorical_cols])

# One-hot encoding
features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

# Normalize features (0-1 range)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
```  

## 🔄 Pipeline Summary  
The model pipeline included pre-tuned AutoML algorithms to enhance prediction without manually optimizing hyperparameters. Here's the pipeline overview:  
1. **Data Preprocessing:** Handled missing data, scaling, and one-hot encoding.  
2. **Model Aggregation:** Leveraged a stack of leading machine learning models:  
   - `CatBoost` for handling categorical data effectively.  
   - `XGBoost` and `LightGBM` for fast, scalable gradient boosting.  
   - A logistic regression model for baseline linear performance.  
3. **Model Selection:** AutoML dynamically selected the best model configuration during cross-validation.  

### Key Pipeline Parameters  
| Model        | Key Parameters                                       | Explanation                                                          |  
|--------------|------------------------------------------------------|----------------------------------------------------------------------|  
| **CatBoost** | num_trees=3000, learning_rate=0.03, max_depth=5      | Efficient with categorical and tabular datasets.                     |  
| **XGBoost**  | booster='gbtree', early_stopping_rounds=30           | Popular for its robustness and high predictive power.                |  
| **LightGBM** | bagging_fraction=0.85, max_depth=-1                  | Designed for speed and scalability with large datasets.              |  
| **Scaling**  | MinMaxScaler for normalization                       | Helps gradient-based models converge faster and prevent dominance.   |  

## 🖥️ Code Highlights  
Here’s a breakdown of the most critical sections of the code:

1. **Data Preprocessing**  
   Converts the raw dataset to a clean format usable by the models.  
   ```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

# Impute missing values and scale features
def transform_data(dataset: pd.DataFrame):
    features = dataset.drop(columns=['NObeyesdad', 'id'])
    target = dataset['NObeyesdad'].values if 'NObeyesdad' in dataset.columns else None

    numeric_imputer = SimpleImputer(strategy='mean')
    features = numeric_imputer.fit_transform(features)

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features, target
   ```  

2. **Model Training and AutoML**  
   Automated the model training process using the **FEDOT AutoML framework** to efficiently explore high-quality predictive models.  
   ```python
from fedot.api.main import Fedot
from fedot.core.repository.tasks import Task, TaskTypesEnum

def train_model(features, target):
    # Configure and train the model
    pipeline = Fedot(problem='classification', preset='best_quality', timeout=1.0, metric='accuracy')
    pipeline.fit(features, target)
    return pipeline
   ```  

3. **Evaluation and Predictions**  
   Evaluated model performance and prepared submission files.  
   ```python
def create_submission():
    predictions = model.predict(test_features)
    submission_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': predictions})
    submission_df.to_csv("submission.csv", index=False)
   ```  

## 📊 Metrics  

The model's performance is evaluated using **accuracy**, which measures the proportion of correct predictions among the total predictions:  
- Accuracy: 90.7%  
This means the model correctly identifies the obesity risk category for 9 out of every 10 individuals in the evaluation set.  

### Interpretation:  
- **Strengths:** High accuracy demonstrates strong predictive capability across multiple classes.  
- **Potential Challenges:** Class imbalance (some obesity categories may have fewer instances) could affect predictions for minority classes.  

## 💡 Takeaways  

This model effectively predicts obesity risk with **90.7% accuracy**, a promising result for public health applications. By leveraging advanced AutoML techniques and robust preprocessing, it demonstrates a scalable, efficient approach to tackle similar classification problems. While this is a synthetic competition dataset, the pipeline could easily be adapted for real-world use cases like predicting cardiovascular risk or targeting dietary interventions.  

Modeling obesity risk is not just about prediction—it's about enabling preventive healthcare measures that could save lives. This competition shows how machine learning can make serious strides in addressing global health challenges.  