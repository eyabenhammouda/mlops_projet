import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from scipy.stats import zscore

def normalize_data(data, columns_to_normalize):
    scaler = MinMaxScaler()
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return data

def drop_columns(data, columns_to_drop):
    return data.drop(columns=columns_to_drop)

def remove_outliers(data, num_cols, method="zscore", threshold=3):
    if method == "zscore":
        z_scores = np.abs(zscore(data[num_cols]))
        data = data[(z_scores < threshold).all(axis=1)]
    elif method == "iqr":
        for col in num_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def prepare_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    data = pd.concat([train_data, test_data], ignore_index=True)
    
    columns_to_drop = ['State', 'Area code', 'Total day minutes',
                       'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    data = drop_columns(data, columns_to_drop)
    
    label_encoder = LabelEncoder()
    data['International plan'] = label_encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = label_encoder.fit_transform(data['Churn'])
    
    numerical_columns = ['Account length', 'Number vmail messages', 'Total day calls', 
                         'Total day charge', 'Total eve calls', 'Total eve charge', 
                         'Total night calls', 'Total night charge', 'Total intl calls', 
                         'Total intl charge', 'Customer service calls']
    
    data = remove_outliers(data, numerical_columns, method="iqr")
    data = normalize_data(data, numerical_columns)
    
    train_data = data.iloc[:len(train_data)]
    test_data = data.iloc[len(train_data):]
    
    X_train = train_data.drop('Churn', axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop('Churn', axis=1)
    y_test = test_data['Churn']
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def optimize_hyperparameters(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def save_model(model, filename):
    joblib.dump(model, filename)

def load_model(filename):
    return joblib.load(filename)
