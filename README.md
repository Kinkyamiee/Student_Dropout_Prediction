### **README: Student Dropout Prediction Project**

---

#### **Project Overview**
The **Student Dropout Prediction Project** aims to develop a machine learning model that predicts the likelihood of a student dropping out of an educational program. This predictive model is designed to help educational institutions identify at-risk students early, enabling timely interventions and support to improve retention rates.

The model is built using a Random Forest Classifier and is deployed via a Streamlit web application, providing a user-friendly interface for educators and administrators to input student data and receive real-time predictions.

---

#### **Table of Contents**
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Performance](#model-performance)
7. [Ethical Considerations](#ethical-considerations)
8. [Future Enhancements](#future-enhancements)
9. [Contributors](#contributors)
10. [License](#license)

---

### **Project Structure**

```
student_dropout_prediction/
│
├── data/ 
│   ├── student_data.csv       # Sample dataset for training and testing
│
├── model/ 
│   ├── best_rf_model.pkl      # Trained Random Forest model
│
├── app.py                    # Streamlit app for model deployment
├── model_training.py          # Script for training the model
├── requirements.txt           # Python dependencies
├── README.md                  # Project README file (this file)
│
└── notebooks/
    ├── data_preprocessing.ipynb  # Jupyter notebook for data cleaning and preprocessing
    ├── exploratory_analysis.ipynb  # Jupyter notebook for EDA and feature selection
    └── model_training.ipynb      # Jupyter notebook for model training and evaluation
```

---

### **Features**
- **Data Preprocessing**: Handles missing values, outliers, and data normalization to prepare student data for training.
- **Random Forest Classifier**: Trained model to predict student dropout based on academic, demographic, and financial features.
- **Streamlit Deployment**: A web application where users can input student data and receive real-time dropout predictions.
- **Feature Importance**: Provides an explanation of key factors driving the dropout prediction, using SHAP values.
  
---

### **Installation**

#### **Step 1**: Clone the Repository
```bash
git clone https://github.com/[your-username]/student_dropout_prediction.git
cd student_dropout_prediction
```

#### **Step 2**: Install Dependencies
The project uses Python 3.7+ and the following dependencies:
```bash
pip install -r requirements.txt
```

**Main Libraries**:
- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning models and utilities.
- `numpy`: Numerical computing.
- `shap`: SHAP (Shapley Additive Explanations) for model interpretability.
- `streamlit`: Deployment platform for the web application.

---

### **Usage**

#### **1. Model Training**
If you want to retrain the model with new data, run the model training script:
```bash
python model_training.py
```
This script will train the Random Forest Classifier and save the best model as a pickle file (`best_rf_model.pkl`) in the `model/` directory.

#### **2. Running the Streamlit App**
To launch the Streamlit web app for real-time predictions:
```bash
streamlit run app.py
```
Once the app is running, you can input student data (e.g., grades, tuition fees, age) and receive predictions on whether the student is at risk of dropping out.

---

### **Model Performance**

The **Random Forest Classifier** was trained and evaluated on a dataset containing student records with the following key features:
- **Curricular units 1st and 2nd sem (approved, grade, evaluations)**
- **Tuition fees up to date**
- **Age at enrollment**
- **Admission grade**
- **Previous qualification grade**

**Model Evaluation Metrics**:
- **Accuracy**: 84%
- **Precision**: 84%
- **Recall**: 68%
- **F1 Score**: 75%

---

### **Ethical Considerations**

The project has taken several ethical considerations into account:
- **Bias Mitigation**: The model was evaluated for potential biases, particularly regarding socio-economic factors like tuition status, to ensure fairness.
- **Student Privacy**: The dataset contains sensitive student information, and care has been taken to anonymize data and follow data protection regulations.
- **Human Oversight**: The model is intended to be a **decision-support tool**. Predictions should not be used in isolation, and human judgment is necessary when making high-stakes decisions about students' futures.

---

### **Future Enhancements**

- **Expanding Dataset**: Integrate additional features such as socio-economic data, psychological factors, and extracurricular involvement for a more holistic model.
- **Automated Retraining**: Implement continuous model retraining based on new data to improve performance over time.
- **Mobile App**: Develop a mobile version of the Streamlit app to make predictions more accessible to educators on the go.

---

### **Contributors**
- **Nwamaka Ajunwa**: Data scientist

---

### **License**
This project is licensed under the **MIT License**. You are free to use, modify, and distribute the project, provided you include attribution to the original authors.

For more details, refer to the LICENSE file in this repository.
