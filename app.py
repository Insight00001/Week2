import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title='School Dropout Dashboard',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='bar_chart'
)
def read_data(data):
    return pd.read_csv(data)

data = read_data('visualize_data.csv')
pca_data =read_data('PCA.csv')
total_data = read_data('total table.csv')
percentage_data = read_data('percentage table.csv')


#-------------Side bar--------------


graph = st.sidebar.selectbox('Select Chart',['Piechart','Barchart','PCA','Grade vs Dropout','Scholarship vs dropout'])
#df_selection = data.query("Gender==@gender & `Scholarship holder`==@Scholarship & Debtor==@debtor & Target==@target")

#--------------main Page ----------------

st.title(':bar_chart: Student Dropout DashBoard')
st.markdown('##')

# calculate KPIs

total_students = data.value_counts().sum()
total_male = data[data['Gender']=='Male'].value_counts().sum()
total_female = data[data['Gender']=='Female'].value_counts().sum()

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

#-------------barplot------------
tp_total_table = total_data.T
tp_total_table = tp_total_table.reset_index()
tp_total_table_=tp_total_table.columns=['Category','Values']

tp_percentage_table = percentage_data.T
tp_percentage_table = tp_percentage_table.reset_index()
tp_percentage_table_=tp_percentage_table.columns=['Category','Percentage']
tp_percentage_table=tp_percentage_table[1:]


dropout_vs_admgrade = data.groupby('Target')['Admission grade'].value_counts().reset_index(name='counts')
dropout_vs_aid = data.groupby('Scholarship holder')['Target'].value_counts().reset_index(name='counts')
if graph =='Barchart':
    st.bar_chart(tp_total_table[1:],x='Category',y='Values',x_label='Category',y_label='Values',height=500)

elif graph=='Piechart':
    fig_ = px.pie(tp_percentage_table, 
             values='Percentage', 
             names='Category',
             title='Percentage Distribution of Total',
             color_discrete_sequence=px.colors.sequential.Plasma)

    # Display the Plotly pie chart in Streamlit
    st.plotly_chart(fig_, use_container_width=True)
elif graph=='Grade vs Dropout':
    fig = px.line(dropout_vs_admgrade, 
                x='Admission grade', 
                y='counts', 
                color='Target',  # Hue variable
                title='Dropout vs Admission Grade',
                labels={'Admission grade': 'Admission Grade', 'counts': 'Counts'})

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
elif graph=='PCA':
    fig = px.scatter(pca_data, 
                 x='PCA1', 
                 y='PCA2', 
                 color='PCA1',# Hue variable for different colors
                 title='2D Scatterplot for PCA',
                 labels={'PCA1': 'X Axis', 'PCA2': 'Y Axis'},
                 color_discrete_sequence=px.colors.sequential.Viridis)

# Display the scatter plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif graph=='Scholarship vs dropout':
    fig = px.bar(dropout_vs_aid, 
             x='Scholarship holder', 
             y='counts', 
             color='Target',  # This is the equivalent of 'hue' in seaborn
             barmode='group',  # Ensures bars for each hue are placed side by side
             title='Effect of Financial Aid on Target',
             labels={'Scholarship holder': 'Scholarship Holder', 'counts': 'Counts'})

# Display the Plotly chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

st.divider()

