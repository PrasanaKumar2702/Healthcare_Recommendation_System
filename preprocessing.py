
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder     .appName("HealthcareRecommendation")     .config("spark.executor.memory", "4g")     .config("spark.driver.memory", "4g")     .getOrCreate()

# Read patient history data (adjust based on your dataset)
data = spark.read.csv("patient_history.csv", header=True, inferSchema=True)

# Convert Spark DataFrame to Pandas for preprocessing
pdf = data.toPandas()

# Preprocessing function to scale data and create input sequences
def preprocess_data(df):
    scaler = MinMaxScaler()
    df['scaled_history'] = scaler.fit_transform(df[['history']].values)
    X = []
    y = []
    for seq in df['scaled_history']:
        X.append(seq[:-1])  # Input
        y.append(seq[-1])   # Label
    X = np.array(X)
    y = np.array(y)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = preprocess_data(pdf)
