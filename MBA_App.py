import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import calendar
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#lets import our data from the AWS RDS MySQL DataBase
#db info
from sqlalchemy import create_engine


host = st.secrets["host"]
user = st.secrets["user"]
password = st.secrets["password"]
port = st.secrets["port"]
database = st.secrets["database"]

st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config( page_title="Market Basket Analysis App",
                    page_icon= "random",
                    layout="wide"
 )



col1, col2, col3 = st.columns((.1,1,.1))

with col1:
    st.write("")

with col2:
    st.markdown(" <h1 style='text-align: center;'>Market Basket Analysis on Online Retail Data:</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><i><b>Providing a Retail Business with a strategy which helps improve their "
                "product sales, inventory management, and customer retention, which in turn would improve the profitability of the business.</b></i></p>", unsafe_allow_html=True)
    st.markdown("<center><img src='https://github.com/kkrusere/Market-Basket-Analysis-on-the-Online-Retail-Data/blob/main/Assets/MBA.jpg?raw=1' width=600/></center>", unsafe_allow_html=True)


with col3:
    st.write("")


st.markdown("----")

col1, col2,col3 = st.columns((1,0.1,1))

with col1:
   
    st.markdown("### ***Project Contributors:***")
    st.markdown("Kuzi Rusere")

    st.markdown("### **Project Introduction**")
    st.markdown("***Business Proposition:*** This project aims to provide a Retail "
                "Business with a strategy that helps improve their product sales, "
                "inventory management, and customer retention, which in turn would "
                "improve the profitability of the business. In the retail environment, "
                "profitability and the `bottom line` is at the focal point of any "
                "organization or business in this space. From product sales, through "
                "inventory management to customer acquisition and retention all this one "
                "way or the other affects the business' profits and net revenue. Transaction "
                "data from the POS (point of sale) systems for a retail business is a treasure "
                "trove of insights that can be used to better understand the products, customer "
                "purchase behavior, and sales together with the relationship and patterns in "
                "between. This project explores the different ways of deriving these insights, "
                "patterns, and relationships, and how these can be used in designing, developing, "
                "and implementing a strategy to improve the retail business' profits, revenue, "
                "and overall operations")
    st.markdown("***Methodology:*** Data Mining, Analysis and Visualization of Retail "
                "Sales Data.")
    """
    1. Market Basket Analysis (MBA), which aims to find relationship and establishing pattens within the retail sales data. <br>
    2. Customer Segmentation 
    > * RFM (recency, frequency, monetary) Analysis
    3. Product Recomendation (people who bought this also bought)
    """
    st.markdown("In addition, we created this `Streamlit` interactive data visualization "
                "tool that allows users interact with the data and analytics.")
with col2:
    pass
with col3:
    st.markdown("### ***Data Collection:***")

    """
    **General Information About the Data**

    This is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

    **Information about the Attributes/Columns in the Dataset**
    * ***InvoiceNo:*** Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
    * ***StockCode:*** Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
    * ***Description:*** Product (item) name. Nominal.
    * ***Quantity:*** The quantities of each product (item) per transaction. Numeric.
    * ***InvoiceDate:*** Invice Date and time. Numeric, the day and time when each transaction was generated.
    * ***UnitPrice:*** Unit price. Numeric, Product price per unit in sterling.
    * ***CustomerID:*** Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
    * ***Country:*** Country name. Nominal, the name of the country where each customer resides.

    ###### **The data source:**
    
    """

    st.image("Assets/UCI_ML_REPO.png", caption="https://archive.ics.uci.edu/ml/datasets/online+retail")


st.markdown("----")

@st.cache(allow_output_mutation=True, ttl= 120.0)
def load_data():
    """
    This fuction loads data from the aws rds mysql table
    """
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online-Retail_Data"
        data = pd.read_sql(query,engine)

        return data

    except Exception as e:
        pass

#loading the data
df = load_data() 

st.markdown("*lets take a look at the data:*")
"""
We are going to use the pandas `.shape` function/method to the total number of columns and rows of the dataframe. We can see that our dataframe contains 541909 rows and 8 columns

We'll use the pandas `.info()` function so see the general infomation (data types, null value count, etc.) about the data.
"""
col1, col2,col3 = st.columns((1, 0.5,.1))

with col1:
    st.markdown("***The below is the first 5 rows of the cleaed dataset***")
    st.dataframe(df.head())
with col2:
    st.markdown("***The below is the shape of the data***")
    st.dataframe(df.shape)

with col3:
    st.markdown("***The below is the info of the data***")
    st.dataframe(df.info())
