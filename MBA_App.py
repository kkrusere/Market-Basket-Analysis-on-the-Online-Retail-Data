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
    #df = pd.read_csv('Cultural_Health Moments_Data.csv')

    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online-Retail_Data"
        data = pd.read_sql(query,engine)

    except Exception as e:
        print(str(e))

    return data
    
df = load_data()