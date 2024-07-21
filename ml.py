import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Load data
data = pd.read_csv('Heart_dataset.csv')

# Drop unnecessary column
df = data.drop(columns=['Unnamed: 0'])

# Map categorical variables
df['Thal'] = df['Thal'].map({'fixed': 1, 'normal': 2, 'reversable': 3})
df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
df['ChestPain'] = df['ChestPain'].map({'typical': 0, 'asymptomatic': 1, 'nonanginal': 2, 'nontypical': 3})

# Fill missing values
df['Thal'] = df['Thal'].fillna(df['Thal'].mode()[0])
df['Ca'] = df['Ca'].fillna(df['Ca'].mean())

# Function to remove outliers
def remove_outliers(df, columns):
    for col in columns:
        q75, q25 = np.percentile(df[col], [75, 25])
        iqr = q75 - q25
        max_val = q75 + (1.5 * iqr)
        min_val = q25 - (1.5 * iqr)
        df.loc[df[col] < min_val, col] = np.nan
        df.loc[df[col] > max_val, col] = np.nan
    df.dropna(inplace=True)
    return df

# Remove outliers
df = remove_outliers(df, ['RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca'])

# Train-test split
x = df.drop('AHD', axis=1)
y = df[['AHD']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Evaluate model
pred = model.predict(x_test)
print("Recall:", recall_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("F1 Score:", f1_score(y_test, pred))
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# Save model
joblib.dump(model, 'heart_disease_model.pkl')


# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
# import joblib

# # Load data
# data = pd.read_csv('Heart_dataset.csv')

# # Drop unnecessary column
# df = data.drop(columns=['Unnamed: 0'])

# # Map categorical variables
# df['Thal'] = df['Thal'].map({'fixed': 1, 'normal': 2, 'reversable': 3})
# df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
# df['ChestPain'] = df['ChestPain'].map({'typical': 0, 'asymptomatic': 1, 'nonanginal': 2, 'nontypical': 3})

# # Fill missing values
# df['Thal'] = df['Thal'].fillna(df['Thal'].mode()[0])
# df['Ca'] = df['Ca'].fillna(df['Ca'].mean())

# # Function to remove outliers
# def remove_outliers(df, columns):
#     for col in columns:
#         q75, q25 = np.percentile(df[col], [75, 25])
#         iqr = q75 - q25
#         max_val = q75 + (1.5 * iqr)
#         min_val = q25 - (1.5 * iqr)
#         df.loc[df[col] < min_val, col] = np.nan
#         df.loc[df[col] > max_val, col] = np.nan
#     df.dropna(inplace=True)
#     return df

# # Remove outliers
# df = remove_outliers(df, ['RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca'])

# # Feature engineering
# df['Age*Chol'] = df['Age'] * df['Chol']
# df['RestBP*MaxHR'] = df['RestBP'] * df['MaxHR']

# # Train-test split
# x = df.drop('AHD', axis=1)
# y = df['AHD']
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

# # Data normalization and model training with pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
#     ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
# ])

# # Hyperparameter tuning
# param_grid = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__max_depth': [None, 10, 20, 30],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(x_train, y_train)

# # Best model evaluation
# best_model = grid_search.best_estimator_
# pred = best_model.predict(x_test)
# print("Recall:", recall_score(y_test, pred))
# print("Precision:", precision_score(y_test, pred))
# print("F1 Score:", f1_score(y_test, pred))
# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))
# print(confusion_matrix(y_test, pred))

# # Save model
# joblib.dump(best_model, 'heart_disease_model.pkl')




# import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
# import joblib

# # Load data
# data = pd.read_csv('Heart_dataset.csv')

# # Drop unnecessary column
# df = data.drop(columns=['Unnamed: 0'])

# # Map categorical variables
# df['Thal'] = df['Thal'].map({'fixed': 1, 'normal': 2, 'reversable': 3})
# df['AHD'] = df['AHD'].map({'No': 0, 'Yes': 1})
# df['ChestPain'] = df['ChestPain'].map({'typical': 0, 'asymptomatic': 1, 'nonanginal': 2, 'nontypical': 3})

# # Fill missing values
# df['Thal'] = df['Thal'].fillna(df['Thal'].mode()[0])
# df['Ca'] = df['Ca'].fillna(df['Ca'].mean())

# # Function to remove outliers
# def remove_outliers(df, columns):
#     for col in columns:
#         q75, q25 = np.percentile(df[col], [75, 25])
#         iqr = q75 - q25
#         max_val = q75 + (1.5 * iqr)
#         min_val = q25 - (1.5 * iqr)
#         df.loc[df[col] < min_val, col] = np.nan
#         df.loc[df[col] > max_val, col] = np.nan
#     df.dropna(inplace=True)
#     return df

# # Remove outliers
# df = remove_outliers(df, ['RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Ca'])

# # Feature engineering
# df['Age*Chol'] = df['Age'] * df['Chol']
# df['RestBP*MaxHR'] = df['RestBP'] * df['MaxHR']

# # Train-test split
# x = df.drop('AHD', axis=1)
# y = df['AHD']
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=42)

# # Data normalization and model training with pipeline
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
#     ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
# ])

# # Hyperparameter tuning
# param_grid = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__max_depth': [None, 10, 20, 30],
#     'classifier__min_samples_split': [2, 5, 10],
#     'classifier__min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(x_train, y_train)

# # Best model evaluation
# best_model = grid_search.best_estimator_

# # Predict on test set
# pred = best_model.predict(x_test)

# # Evaluation metrics
# print("Recall:", recall_score(y_test, pred))
# print("Precision:", precision_score(y_test, pred))
# print("F1 Score:", f1_score(y_test, pred))
# print("Accuracy:", accuracy_score(y_test, pred))
# print(classification_report(y_test, pred))
# print(confusion_matrix(y_test, pred))
# print(df)
# # Save model
# joblib.dump(best_model, 'heart_disease_model.pkl')
