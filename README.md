
# Personalized Healthcare Recommendation System

This project demonstrates a healthcare recommendation system using Deep Learning models (RNNs and LSTMs) built with TensorFlow. The goal is to predict patient treatment adherence or suggest personalized treatment plans based on their historical medical records.

## Features
- **RNN/LSTM Model:** Predict next steps in treatment based on patient history.
- **Apache Spark Integration:** For distributed computing on large datasets.
- **TensorFlow for Deep Learning:** Model training and evaluation.
  
## Folder Structure
- `preprocessing.py`: Data preprocessing and Spark setup for large datasets.
- `model.py`: LSTM model definition, training, and evaluation.
- `predict.py`: Code for making predictions based on the trained model.
- `README.md`: Project documentation.

## Instructions
1. Install dependencies:
    ```bash
    pip install tensorflow pyspark pandas numpy scikit-learn
    ```
2. Preprocess the data and train the model:
    ```bash
    python model.py
    ```
3. Make predictions:
    ```bash
    python predict.py
    ```

## Dependencies
- Python 3.8+
- TensorFlow
- PySpark
- NumPy
- Pandas
- Scikit-learn
