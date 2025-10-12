import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    print("Loading Boston Housing dataset...")
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df


def preprocess_data(df, target_column='MEDV', test_size=0.2, random_state=42):
    print("Preprocessing data...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Train set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(model, X_train, y_train):
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print("Training completed!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"{model_name} Results:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    return mse


def display_dataset_info(df):
    print("=" * 60)
    print("BOSTON HOUSING DATASET INFORMATION")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()[:-1]}")
    print(f"Target variable: {df.columns[-1]}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    print("=" * 60)


def run_complete_pipeline(model, model_name="Model"):
    df = load_data()
    display_dataset_info(df)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    trained_model = train_model(model, X_train, y_train)
    mse = evaluate_model(trained_model, X_test, y_test, model_name)
    return mse
