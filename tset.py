import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load data (replace with your data)
# Example Data
st.title('PCA Dashboard Summary')

# Upload Dataset
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset:")
    st.dataframe(data.head())

    # Display summary statistics
    st.write("### Summary Statistics:")
    st.write(data.describe())

    # Display the PCA plot
    st.write("### PCA Analysis")
    
    # Sidebar for PCA options
    st.sidebar.header("PCA Options")
    n_components = st.sidebar.slider("Number of PCA components", 2, min(len(data.columns), 10))

    # Perform PCA if the dataset is valid
    if st.sidebar.button("Perform PCA"):
        # Standardize the data if needed
        X = data.select_dtypes(include=[np.number])  # Select only numerical columns for PCA

        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_components = pca.fit_transform(X)

        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(n_components)])

        # Optionally, add the target column if present
        if 'Target' in data.columns:
            pca_df['Target'] = data['Target']
        
        # Display the explained variance ratio
        st.write("### Explained Variance Ratio:")
        st.write(pca.explained_variance_ratio_)

        # Plot the PCA pairplot
        st.write("### PCA Pairplot:")
        fig = sns.pairplot(pca_df, hue="Target" if 'Target' in data.columns else None)
        st.pyplot(fig)

        # Plot a scree plot of the explained variance ratio
        st.write("### Scree Plot:")
        fig, ax = plt.subplots()
        ax.plot(np.arange(1, n_components + 1), pca.explained_variance_ratio_, marker='o')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Explained')
        st.pyplot(fig)

    # Add more charts/plots here if needed
    st.write("### Correlation Heatmap")
    numeric_data = data.select_dtypes(['int','float'])
    fig, ax = plt.subplots(figsize=(12,12))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write('Percentage')
    per_table = pd.read_csv('percentage table.csv')
    label=per_table.columns
    values=per_table.values
    explode=(0, 0.1, 0, 0)
    fig_,ax_ =plt.subplots(figsize=(12,12))
    ax_.pie(values,labels=label,explode=explode,autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax_.axis('equal')
    # ax.tick_params(rotation=90)
    # plt.tight_layout()
    st.pyplot(fig_)

else:
    st.write("Please upload a dataset to continue.")

