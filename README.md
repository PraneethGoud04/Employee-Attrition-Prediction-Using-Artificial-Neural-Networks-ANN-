# Employee-Attrition-Prediction-Using-Artificial-Neural-Networks-ANN-


## 📌 Project Overview

Employee attrition is a major challenge for organizations because employee turnover can increase recruitment costs and affect productivity. This project uses an Artificial Neural Network (ANN) to predict whether an employee is likely to stay with or leave an organization based on employee profile, job satisfaction, compensation, work-life balance, and other workplace-related factors.

The project includes data analysis, feature engineering, categorical encoding, feature scaling, ANN model training, hyperparameter tuning, and an interactive Streamlit application for predicting attrition risk.

## 🎯 Objectives

- Analyze employee data to identify patterns associated with attrition.
- Detect and handle skewness in numerical features.
- Perform feature engineering to derive additional employee-related indicators.
- Build an ANN model for binary classification.
- Optimize model hyperparameters using Optuna.
- Develop an interactive application for employee attrition risk prediction.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow / Keras
- **Hyperparameter Optimization:** Optuna
- **Web Application:** Streamlit
- **Model Persistence:** Pickle and Keras model saving

## 📊 Dataset Features

The dataset contains 100,000 employee records and 18 original columns, including:

- Age
- Gender
- Years at Company
- Job Level
- Job Role
- Monthly Income
- Work-Life Balance
- Job Satisfaction
- Performance Rating
- Overtime
- Distance from Home
- Education Level
- Marital Status
- Number of Dependents
- Company Tenure
- Employee Recognition
- Promotions
- Attrition (Target Variable)

After feature engineering, additional features are created to help the model learn potentially useful relationships.

## 🔍 Project Workflow

### 1. Exploratory Data Analysis (EDA)

- Examined dataset dimensions, data types, and statistical summaries.
- Checked for missing values and duplicate records.
- Analyzed attrition class distribution.
- Used boxplots to identify potential outliers.

### 2. Data Preprocessing

- Applied logarithmic transformation to Monthly Income using `np.log1p()`.
- Encoded nominal and ordinal categorical variables using One-Hot Encoding and Ordinal Encoding.
- Applied StandardScaler to numerical features.
- Combined preprocessing steps using Scikit-learn's `ColumnTransformer`.

### 3. Feature Engineering

Created the following additional features:

- **Monthly Income Log:** Reduces income skewness.
- **Income per Year:** Monthly Income Log divided by Years at Company plus one.
- **Promotion Rate:** Number of Promotions divided by Years at Company plus one.
- **Work Stress:** Derived from overtime and poor or fair work-life balance.

### 4. ANN Model Development

Built a feed-forward Artificial Neural Network using TensorFlow/Keras.

The baseline architecture contains:
- Input layer
- Two hidden Dense layers with ReLU activation
- Output layer with Sigmoid activation

The model uses the Adam optimizer and Binary Cross-Entropy loss for binary classification.

### 5. Hyperparameter Tuning

Used Optuna to explore hyperparameters such as:

- Number of hidden layers
- Number of neurons
- Activation function
- Optimizer
- Learning rate
- Dropout rate
- L1 regularization strength

The final model also uses Batch Normalization, Dropout, L1 regularization, and Early Stopping to help control overfitting and improve training.

### 6. Model Evaluation

Evaluated model performance using:

- Accuracy
- Precision
- Recall
- F1-score
- Training and validation accuracy curves

The final model achieved approximately **89% test accuracy** in the notebook evaluation. Actual performance may vary with retraining and data changes.

### 7. Streamlit Application

Developed an interactive web interface that allows users to enter employee details and obtain an attrition prediction.

The application:
1. Collects employee information.
2. Performs the same feature engineering used during training.
3. Applies the saved preprocessing pipeline.
4. Generates an attrition probability using the trained ANN.
5. Displays a risk estimate and a predicted stay/leave classification.

## 📈 Model Performance

The final ANN achieved approximately 89% test accuracy on the held-out test set.

Accuracy alone does not fully describe classification performance, so precision, recall, and F1-score are also considered to assess predictions for both attrition classes.

## 📁 Project Structure

```text
employee-attrition-ann/
│
├── app.py
├── ANN_EMPLOYEE_ATTRITION_PROJECT.ipynb
├── employee_attrition_model.keras
├── data_preprocessing.pkl
├── requirements.txt
└── README.md
```

*Adjust the filenames to match the files in your repository.*

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd employee-attrition-ann
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

