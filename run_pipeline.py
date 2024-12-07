import pandas as pd
import numpy as np
from scripts.preprocess_data import data_preprocessing
from scripts.model_dev import model_dev
def process_post(path):
    return pd.read_csv(path)


if __name__ == '__main__':
    data = process_post("D:\FREELANCE_PROJECTS\Mental-Health-Prediction-ML\data\survey.csv")
    cleaned_data = data_preprocessing(data=data)
    models_accuracy = model_dev(filtered_data=cleaned_data)
    print(models_accuracy)


