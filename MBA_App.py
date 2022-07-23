import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

st.markdown("### ***Project Contributors:***")
st.markdown("Kuzi Rusere")

@st.cache(allow_output_mutation=True, ttl= 120.0)
def load_data():
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online-Retail_Data"
        data = pd.read_sql(query,engine)

        return data

    except Exception as e:
        pass

    
df = load_data()
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