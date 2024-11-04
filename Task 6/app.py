import os
import pickle
import streamlit as st

# Print the current working directory
print("Current working directory:", os.getcwd())

# Load the trained model
model_path = r"C:\Users\hp\Documents\3Signet Internship\Task 6\best_rf_model.pkl"

with open(model_path, 'rb') as file:
    classifier = pickle.load(file)

# Define mapping for 'Age at enrollment' and 'Tuition fees up to date'
age_enrollment_mapping = {
    "18-24": 0,
    "25-34": 1,
    "35-44": 2,
    "45+": 3
}

tuition_fees_mapping = {
    "Up to date": 0,
    "Not up to date": 1
}

# Defining the function to make predictions using the user input
@st.cache_data()
def prediction(curricular_units_2nd_approved, curricular_units_2nd_grade, curricular_units_1st_approved,
               curricular_units_1st_grade, tuition_fees_status, age_enrollment, admission_grade, 
               previous_qualification_grade, curricular_units_2nd_evaluations):

    # Map 'Age at enrollment' and 'Tuition fees up to date' to numerical values
    age_enrollment_numeric = age_enrollment_mapping[age_enrollment]
    tuition_fees_numeric = tuition_fees_mapping[tuition_fees_status]

    # Making Predictions
    prediction = classifier.predict([[curricular_units_2nd_approved, curricular_units_2nd_grade,
                                      curricular_units_1st_approved, curricular_units_1st_grade,
                                      tuition_fees_numeric, age_enrollment_numeric, admission_grade,
                                      previous_qualification_grade, curricular_units_2nd_evaluations]])

    if prediction == 0:
        pred = "Not Dropout"
    else:
        pred = "Dropout"

    return pred

# Main function to define the Streamlit web app
def main():
    # Front end elements of the web page
    html_temp = '''
    <div style='background-color: blue; padding:13px'>
    <h1 style='color: black; text-align: center;'>Student Dropout Prediction ML App</h1>
    </div>
    '''

    # Display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for user data
    curricular_units_2nd_approved = st.number_input("Curricular units 2nd sem (approved)")
    curricular_units_2nd_grade = st.number_input("Curricular units 2nd sem (grade)")
    curricular_units_1st_approved = st.number_input("Curricular units 1st sem (approved)")
    curricular_units_1st_grade = st.number_input("Curricular units 1st sem (grade)")
    tuition_fees_status = st.selectbox('Tuition fees up to date', tuple(tuition_fees_mapping.keys()))
    age_enrollment = st.selectbox('Age at enrollment', tuple(age_enrollment_mapping.keys()))
    admission_grade = st.number_input("Admission grade")
    previous_qualification_grade = st.number_input("Previous qualification (grade)")
    curricular_units_2nd_evaluations = st.number_input('Curricular units 2nd sem (evaluations)')

    result = ""

    # When 'Predict' is clicked, make prediction and display the result
    if st.button("Predict"):
        result = prediction(curricular_units_2nd_approved, curricular_units_2nd_grade, curricular_units_1st_approved,
                            curricular_units_1st_grade, tuition_fees_status, age_enrollment, admission_grade,
                            previous_qualification_grade, curricular_units_2nd_evaluations)
        st.success("Prediction: {}".format(result))

# Ensure proper entry point for the app
if __name__ == '__main__':
    main()
