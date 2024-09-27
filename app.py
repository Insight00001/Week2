import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import chi2_contingency
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
st.set_page_config(
    page_title="Student Dropout dashboard",
    page_icon="bar_chart",
    layout='wide'
)
st.title('Exploratory Data Analysis for Student dropout')
def Categorical_columns(data):
    categorical_columns =['Marital status','Application mode','Application order',"Daytime/evening attendance\t",
                      'Previous qualification','Nacionality','Debtor','Tuition fees up to date', 'Gender', 'Scholarship holder',
                      'International','Curricular units 1st sem (credited)','Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
                       'Curricular units 1st sem (approved)','Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)','Curricular units 2nd sem (evaluations)',
                       'Curricular units 2nd sem (approved)','Curricular units 2nd sem (without evaluations)',"Mother's occupation","Father's occupation",'Displaced',
                       'Educational special needs','Target']

    for  cat in categorical_columns:
        data[cat] = pd.Categorical(data[cat])
def map_data(data):
    mappings = {'Displaced': {1: 'Yes', 0: 'No'},
            'Tuition fees up to date': {1: 'Yes', 0: 'No'},
           'Gender':{1:'Male',0:'Female'},
            'Debtor':{1:'Yes',0:'No'},
            'Scholarship holder':{1:'Yes',0:'No'},
            'International':{1:'Yes',0:'No'},
            "Daytime/evening attendance\t":{1:'Yes',0:'No'}
           }

# Apply the mapping
    for column, mapping in mappings.items():
        data[column] = data[column].map(mapping)

def filter_categorical(data):
    cat_data=data.select_dtypes(['category'])
    return cat_data

def filter_numeric(data):
    numeric_data = data.select_dtypes(['float','int'])
    return numeric_data


def preprocess_data():
    data =pd.read('cleaned_data.csv')

    mapping =  {'Graduate':2,'Dropout':1,'Enrolled':0}
    data['Target'] = data['Target'].map(mapping)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

df = pd.read_csv('cleaned_data.csv')
#df['Daytime/evening attendance\t'] = df['Daytime/evening attendance\t'].str.replace('\t', '')
total_students = df.value_counts().sum()
total_male = df[df['Gender']==1].value_counts().sum()
total_female = df[df['Gender']==0].value_counts().sum()

col1,col2,col3=st.columns(3)
with col1:
    st.subheader('Total students')
    st.subheader(total_students)
with col2:
    st.subheader('Total female')
    st.subheader(total_female)
with col3:
    st.subheader('Total male')
    st.subheader(total_male)

st.divider()

Categorical_columns(df)
map_data(df)
st.subheader('Dataset Preview')
st.write(df.head())

st.subheader('Data Summary for Categorical')
data_cat = filter_categorical(df)
st.write(data_cat.describe())

st.subheader('Data Summary for Numeric Data')
data_numeric = filter_numeric(df)
st.write(data_numeric.describe())


st.subheader('Filter Categorical Columns')

cat_column = data_cat.columns.to_list()

selected_column_cat = st.selectbox('Select categorical features',cat_column)
unique_value_cat = data_cat[selected_column_cat].unique()
st.subheader('Unique value')
selected_value_cat=st.selectbox("Selected Values",unique_value_cat)
filtered_df_cat = df[df[selected_column_cat]==selected_value_cat]
st.write(filtered_df_cat)

st.subheader('Filter numeric Columns')
num_column = data_numeric.columns.to_list()
selected_column_num = st.selectbox('Select Numeric Features',num_column)
unique_value_num = data_numeric[selected_column_num].unique()
st.subheader('Unique Value')
selected_value_num = st.selectbox('Selected Value',unique_value_num)
filtered_df_num = df[df[selected_column_num]==selected_value_num]
st.write(filtered_df_num)

st.subheader('Plot')
x_column = st.selectbox('select x-axis',df.columns)
#y_column = st.selectbox('select y-axis',cat_column)
if st.button('Generate Plot'):
    if x_column in data_cat:
        fig=px.bar(df, x=x_column,title=f"Distribution of {x_column}")
    

# st.subheader('Plot Numerical Columns')
# if st.button('Generate Plot'):
    elif x_column in data_numeric:
        fig = px.histogram(df,x=x_column,title=f"Distribution of {x_column}")
    st.plotly_chart(fig)
        

st.subheader('Bivariate Analysis')
x_column_ = st.selectbox('Select Categorical column',data_cat.columns)
y_column_ = st.selectbox('Select Numeric column',data_numeric.columns)
if st.button('Generate'):
    if x_column_ and y_column_ in data_numeric:
        fig = px.box(df,x=x_column_,y=y_column_,title=f"Box Plot of {y_column_} by {x_column_}")
    st.plotly_chart(fig)
st.subheader('Correlation matrix for Numeric Data')
fig = px.imshow(data_numeric.corr(),text_auto=True,
                aspect='auto', title="Correlation Heatmap of Numeric Data",
                color_continuous_scale='Blues',height=600,width=1500)
st.plotly_chart(fig)
def percentage_display():
    percentage_data = pd.read_csv('percentage table.csv')   
    tp_percentage_table = percentage_data.T
    tp_percentage_table = tp_percentage_table.reset_index()
    tp_percentage_table_=tp_percentage_table.columns=['Category','Percentage']
    tp_percentage_table=tp_percentage_table[1:]
    return tp_percentage_table
pt =percentage_display()
fig_ = px.pie(pt, 
             values='Percentage', 
             names='Category',
             title='Percentage Distribution of Total',
             color_discrete_sequence=px.colors.sequential.Plasma)

    # Display the Plotly pie chart in Streamlit
st.plotly_chart(fig_, use_container_width=True)
st.divider()
st.subheader('Principal Component Analysis')
pca_number = st.slider('Select Number of Component',min_value=2,max_value=4)


pca_data= pd.read_csv('cleaned_data.csv')

mapping ={'Dropout':0,'Graduate':1,'Enrolled':2}
pca_data['Target'] = pca_data['Target'].map(mapping)

scaler = StandardScaler()
scaled_data=scaler.fit_transform(pca_data)

pca = PCA(n_components=pca_number)
pca_result = pca.fit_transform(scaled_data)
pca_df_2 = pd.read_csv('PCA.csv')
pca_df_3 = pd.read_csv('PCA3.csv')
pca_df_4 = pd.read_csv('PCA4.csv')
if pca_number==2:
    fig = px.scatter(pca_df_2,x='PCA1',y='PCA2',color=pca_data['Target'],
                         color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig)
elif pca_number==3:
    fig= px.scatter_3d(pca_df_3,x='PCA1',y='PCA2',z='PCA3',
                            color=pca_data['Target'],
                         color_discrete_sequence=px.colors.qualitative.Set1)
    
    st.plotly_chart(fig)
elif pca_number==4:
    fig= px.scatter_3d(pca_df_4,x='PCA1',y='PCA2',z='PCA3',
                            color=pca_df_4['PCA4'],
                         color_discrete_sequence=px.colors.qualitative.Set1)
    st.plotly_chart(fig)

st.subheader('Hypothesis testing')
st.write('H0: The Categorical variable does not affect the Target')
st.write("H1: The categorical variable affect the Target")
def chisquaretest(data,var1,var2):
    contingency_table = pd.crosstab(data[var1],data[var2])
    chi,p_value,dof, expectation = chi2_contingency(contingency_table)
    if p_value<0.05:
        st.write(f'The variables {var1} and {var2} are dependent reject H0')
    else:
        st.write(f'The variable {var1} and {var2} are independent fail to reject H0')
    pass

test_category=st.selectbox('Select features',data_cat.columns)

target_class = 'Target'
chisquaretest(data_cat,test_category,target_class)
