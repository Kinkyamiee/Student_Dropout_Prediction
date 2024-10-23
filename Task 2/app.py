import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set up custom CSS for a beautiful background
page_bg_img = '''
<style>
    body {
        background-image: url("https://www.shutterstock.com/image-illustration/studio-backdrop-wallpaper-inside-room-260nw-1921213640.jpg");
       background-size: cover;
    }
    .stApp {
        background: rgba(255, 255, 255, 0.9);  /* Optional: Light background on content */
    }
</style>
'''
# Apply the background image using custom HTML
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(r'C:\Users\hp\Documents\Task 1\students_dropout.csv')
    return data

df = load_data()

# Sidebar for filters
st.sidebar.header('Filter Options')

# Dropdown (selectbox) for Target filter with 'None' option to cancel selection
selected_target = st.sidebar.selectbox('Select Target:', ['None'] + list(df['Target'].unique()))

# Apply filters based on the dropdown selections
filtered_data = df.copy()

if selected_target != 'None':
    filtered_data = filtered_data[filtered_data['Target'] == selected_target]

# Dashboard visuals
st.title('Interactive Data Dashboard')

# PCA Analysis and Visualization
# Select numerical columns for PCA
numerical_columns = ['Admission grade', 
                     'Curricular units 1st sem (grade)', 
                     'Curricular units 2nd sem (grade)',
                     'GDP',
                     'Curricular units 1st sem (approved)', 
                     'Tuition fees up to date',
                     'Age at enrollment', 
                     'Curricular units 2nd sem (evaluations)']

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[numerical_columns])

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 components for 2D visualization
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame for PCA results
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['Target'] = df['Target']  # Add the 'Target' column for coloring

# Plot the PCA results as a scatter plot
st.subheader('PCA Scatter Plot of Numerical Variables')
fig_pca_scatter = px.scatter(pca_df, x='PC1', y='PC2', color='Target', 
                             title='PCA of Numerical Variables',
                             labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} Variance)',
                                     'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} Variance)'})

# Display the PCA scatter plot in Streamlit
st.plotly_chart(fig_pca_scatter)

# Explained variance of each component
explained_variance = pca.explained_variance_ratio_
explained_variance

# Plot Admission Grade vs Target
st.subheader('Bar Chart: Admission Grade vs Target')
fig_bar = px.bar(filtered_data, x='Admission grade', y='Target', 
                 color='Target', title="Admission Grade vs Target")
st.plotly_chart(fig_bar)

# Scatter Plot: Admission Grade vs Curricular Units 1st Sem (Grade)
st.subheader('Scatter Plot: Admission Grade vs Curricular Units 1st Sem (Grade)')
fig_scatter = px.scatter(filtered_data, x='Admission grade', y='Curricular units 1st sem (grade)', 
                         color='Target', title="Admission Grade vs Curricular Units 1st Sem (Grade)")
st.plotly_chart(fig_scatter)
# Correlation Heatmap
numeric_columns = ['Admission grade', 
                    'Curricular units 1st sem (grade)', 
                    'Curricular units 2nd sem (grade)',
                    'GDP',
                    'Curricular units 1st sem (approved)', 
                    'Tuition fees up to date',
                    'Age at enrollment', 
                    'Curricular units 2nd sem (evaluations)']
filtered_numeric_data = filtered_data[numeric_columns]

st.subheader('Correlation Heatmap')
fig, ax = plt.subplots()
sns.heatmap(filtered_numeric_data.corr(), annot=True, ax=ax)
st.pyplot(fig)

# Display filtered data
st.subheader('Filtered Data')
st.write(filtered_data)