# School Dropout Dashboard

This is a **Streamlit** web application designed to visualize and analyze data related to school dropout rates. The dashboard provides insights into various factors affecting dropout rates such as gender, financial aid, and admission grades. The app also includes a PCA (Principal Component Analysis) visualization for deeper data exploration.

## Features

- **Total Students Dashboard**: Displays key performance indicators (KPIs) such as the total number of students, male students, and female students.
- **Chart Visualizations**:
  - **Pie Chart**: Displays the percentage distribution of total student data.
  - **Bar Chart**: Visualizes the total values for different categories.
  - **Dropout vs Admission Grade**: Shows how admission grades relate to dropout rates.
  - **Scholarship vs Dropout**: Analyzes the effect of financial aid on dropout rates.
  - **PCA Plot**: 2D scatterplot for Principal Component Analysis (PCA).
  
## Installation

### Prerequisites

You need to have Python installed, as well as the following Python libraries:

- `streamlit`
- `pandas`
- `plotly`

You can install the required libraries using `pip`:

```bash
pip install streamlit pandas plotly
 
# SetUp
git clone https://github.com/your-username/your-repo-name.git

# Navigate into the project directory:
cd your-repo-name

# run the app
streamlit run app.py


# File Descriptions

    app.py: The main Streamlit app script, responsible for rendering the dashboard and all charts.
    visualize_data.csv: Data file containing the primary dataset.
    PCA.csv: Data file containing the PCA-transformed dataset.
    total table.csv: Dataset for total student counts used in bar chart visualizations.
    percentage table.csv: Dataset for percentage values used in pie chart visualizations.

# Usage

When running the app, you will be able to select different visualizations from the sidebar:

    Piechart: Displays the percentage distribution of total students.
    Barchart: Visualizes total student counts by category.
    Grade vs Dropout: Line chart showing the relationship between admission grades and dropout rates.
    Scholarship vs Dropout: Bar chart that displays how financial aid impacts dropout rates.
    PCA: Scatterplot showing the results of Principal Component Analysis.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Acknowledgments

This app uses the following libraries:

    Streamlit
    Pandas
    Plotly