import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


model = joblib.load("model/RandomForestClassifier.joblib")


def handle_gender(gender):
    return 1 if gender.lower() == 'male' else 0

def handle_no_employees(no_employees):
    mapping = {
        '1-5': 0,
        '6-25': 1,
        '26-100': 2,
        '100-500': 3,
        '500-1000': 4,
        'More than 1000': 5
    }
    return mapping.get(no_employees, 0)

def handle_mental_health_consequence(value):
    return {'No': 0, 'Yes': 1, 'Maybe': 2}.get(value, 2)

self_employed_mapping = {'No': 0, 'Yes': 1}
family_history_mapping = {'No': 0, 'Yes': 1}
work_interfere_mapping = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3}
tech_company_mapping = {'No': 0, 'Yes': 1}
benefits_mapping = {'No': 0, "Don't know": 1, 'Yes': 2}
remote_work_mapping = {'No': 0, 'Yes': 1}
leave_mapping = {
    'Very easy': 4,
    'Somewhat easy': 3,
    'Somewhat difficult': 2,
    'Very difficult': 1,
    "Don't know": 0
}

st.set_page_config(page_title="Mental Health Prediction App", layout="centered")
st.title("Mental Health Prediction")

st.markdown(
    """
    ### Input the details below to predict mental health issues and explore insights.
    """
)

age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", options=["Female", "Male"])
self_employed = st.selectbox("Self-Employed?", options=["Yes", "No"])
family_history = st.selectbox("Family History of Mental Health Issues?", options=["Yes", "No"])
work_interfere = st.selectbox("Work Interference Level", options=["Never", "Rarely", "Sometimes", "Often"])
no_employees = st.selectbox("Number of Employees", options=["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"])
tech_company = st.selectbox("Works in a Tech Company?", options=["Yes", "No"])
benefits = st.selectbox("Mental Health Benefits Provided?", options=["Yes", "No", "Don't know"])
leave = st.selectbox("Ease of Taking Leave for Mental Health Reasons", options=["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"])
remote_work = st.selectbox("Works Remotely?", options=["Yes", "No"])
mental_health_consequence = st.selectbox("Afraid of Mental Health Consequences?", options=["Yes", "No", "Maybe"])
phys_health_consequence = st.selectbox("Afraid of Physical Health Consequences?", options=["Yes", "No", "Maybe"])
mental_health_interview = st.selectbox("Willing to Discuss Mental Health in Interviews?", options=["Yes", "No", "Maybe"])

input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "self_employed": self_employed,
    "family_history": family_history,
    "work_interfere": work_interfere,
    "no_employees": no_employees,
    "tech_company": tech_company,
    "benefits": benefits,
    "leave": leave,
    "remote_work": remote_work,
    "mental_health_consequence": mental_health_consequence,
    "phys_health_consequence": phys_health_consequence,
    "mental_health_interview": mental_health_interview
}])

input_data["Gender"] = input_data["Gender"].apply(handle_gender)
input_data["no_employees"] = input_data["no_employees"].apply(handle_no_employees)
input_data["mental_health_consequence"] = input_data["mental_health_consequence"].apply(handle_mental_health_consequence)
input_data["phys_health_consequence"] = input_data["phys_health_consequence"].apply(handle_mental_health_consequence)
input_data["mental_health_interview"] = input_data["mental_health_interview"].apply(handle_mental_health_consequence)
input_data['self_employed'] = input_data['self_employed'].map(self_employed_mapping)
input_data['family_history'] = input_data['family_history'].map(family_history_mapping)
input_data['work_interfere'] = input_data['work_interfere'].map(work_interfere_mapping)
input_data['tech_company'] = input_data['tech_company'].map(tech_company_mapping)
input_data['benefits'] = input_data['benefits'].map(benefits_mapping)
input_data['remote_work'] = input_data['remote_work'].map(remote_work_mapping)
input_data['leave'] = input_data['leave'].map(leave_mapping)

st.write("### Input Data Preview")
st.dataframe(input_data)

if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        result = "Likely to face Mental Health Issues" if prediction[0] == 1 else "Unlikely to face Mental Health Issues"
        
        if prediction[0] == 1:
            st.error(f"Prediction: {result}")
            st.write("""
            #### Recommendations:
            - **Consider Therapy**: Seek professional therapy or counseling to address mental health challenges.
            - **Mindfulness Practices**: Try mindfulness techniques such as meditation or deep breathing exercises.
            - **Support Networks**: Build or strengthen your support network by connecting with family, friends, or support groups.
            - **Lifestyle Changes**: Regular physical activity, a balanced diet, and adequate sleep can help improve mental health.
            """)
        else:
            st.success(f"Prediction: {result}")
            st.write("""
            #### Recommendations for Maintaining Good Mental Health:
            - **Stay Active**: Engage in physical activities like walking, yoga, or sports to boost mental well-being.
            - **Social Interaction**: Stay connected with loved ones to maintain a positive social environment.
            - **Stress Management**: Use techniques like journaling or time management to keep stress levels under control.
            - **Hobbies**: Pursue hobbies or activities you enjoy to keep your mind engaged and fulfilled.
            """)

        st.write("### Visual Analysis")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.barplot(x=input_data.columns, y=input_data.iloc[0], ax=ax[0])
        ax[0].set_title("Feature Contribution")
        ax[0].tick_params(axis='x', rotation=90)
        
        ax[1].pie([prediction[0], 1 - prediction[0]], labels=["Yes", "No"], autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
        ax[1].set_title("Prediction Outcome")
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

