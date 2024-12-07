import pandas as pd
import numpy as np



def handle_gender(gender):
    if gender == 'Male':
        return 1
    elif gender == 'male':
        return 1
    elif gender == 'female':
        return 0
    else:
        return 0


def handle_no_employees(no_employees):
    if no_employees == '1-5':
        return 0
    elif no_employees == '6-25':
        return 1
    elif no_employees == '26-100':
        return 2
    elif no_employees == '500-1000':
        return 3    
    elif no_employees == 'More than 1000':
        return 4 

def handle_mental_health_consequence(mental_health_consequence):
    if mental_health_consequence == 'No':
        return 0
    elif mental_health_consequence == 'Yes':
        return 1
    else:
        return 2


def data_preprocessing(data):
    try:
        important_columns = [
            "Age", "Gender", "self_employed", "family_history", "work_interfere",
            "no_employees", "tech_company", "benefits",
            "leave","treatment", "remote_work", "mental_health_consequence", "phys_health_consequence", "mental_health_interview"
        ]

        filtered_data = data[important_columns]
        filtered_data["work_interfere"] = filtered_data["work_interfere"].fillna("Sometimes")
        filtered_data["self_employed"] = filtered_data["self_employed"].fillna("No")
        categorical_columns = filtered_data.select_dtypes(include=["object"]).columns
        numeric_columns = filtered_data.select_dtypes(include=["int64", "float64"]).columns
        filtered_data["Gender"] = filtered_data["Gender"].apply(handle_gender)
        filtered_data["no_employees"] = filtered_data["no_employees"].apply(handle_gender)
        filtered_data["mental_health_consequence"] = filtered_data["mental_health_consequence"].apply(handle_mental_health_consequence)
        filtered_data["phys_health_consequence"] = filtered_data["phys_health_consequence"].apply(handle_mental_health_consequence)
        filtered_data["mental_health_interview"] = filtered_data["mental_health_interview"].apply(handle_mental_health_consequence)

        self_employed_mapping = {'No': 0, 'Yes': 1}
        family_history_mapping = {'No': 0, 'Yes': 1}
        work_interfere_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
        tech_company_mapping = {'No': 0, 'Yes': 1}
        benefits_mapping = {'No': 0, "Don't know": 1, 'Yes': 2}
        remote_work_mapping = {'No': 0, 'Yes': 1}
        leave_mapping = {'Somewhat easy': 0, "Don't know": 1, 'Somewhat difficult': 2, 'Very difficult':3, 
        'Very easy': 4}

        filtered_data['self_employed'] = filtered_data['self_employed'].map(self_employed_mapping)
        filtered_data['family_history'] = filtered_data['family_history'].map(family_history_mapping)
        filtered_data['work_interfere'] = filtered_data['work_interfere'].map(work_interfere_mapping)
        filtered_data['tech_company'] = filtered_data['tech_company'].map(tech_company_mapping)
        filtered_data['benefits'] = filtered_data['benefits'].map(benefits_mapping)
        filtered_data['remote_work'] = filtered_data['remote_work'].map(remote_work_mapping)
        filtered_data['leave'] = filtered_data['leave'].map(leave_mapping)

        return filtered_data

    except Exception as e:
        raise e
