from matplotlib.backends.backend_agg import RendererAgg
import streamlit as st
import xmltodict
from pandas import json_normalize
import urllib.request
import seaborn as sns
import matplotlib
from matplotlib.figure import Figure
from PIL import Image
import gender_guesser.detector as gender
from streamlit_lottie import st_lottie
import requests
import random
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st

import pandas as pd
import numpy as np
import chart_studio.plotly as py
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt 
import seaborn as sns
import calendar
import datetime as dt
from wordcloud import WordCloud
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend as ml
from IPython import display
from IPython.core.display import display, HTML
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

st.set_page_config(layout="wide")




st.title("DS-620 Market Basket Analysis and Recency, Frequency, Monetary Analysis Project")
st.markdown("### ***Project Contributors:***")
st.markdown("Supriya Teegala, Manideep Lenkalapally, Mohamad Sahil & Kuzi Rusere")
st.markdown("### ***Business Proposition:***")
st.markdown("Providing a Retail Business with a strategy which helps improve their product sales, inventory management, and customer retention, hence improving the profitability of the business.")
st.markdown("### ***Methodology:***")
st.markdown("Data Mining, Analysis and Visualization of Retail Sales Data." 
            "This will be done mainly using Market Basket Analysis (MBA), "
            "which aims to find relationship and establishing pattens within "
            "the retail sales data or purchases. MBA looks for relationships or "
            "associations among entities and objects that frequently appear together "
            "(for example in a retail sales dataset), such as the collection of items in a shopper’s cart."
            "We are also going to be using Recency, Frequency, Monetary Analysis, which is a analytics technique "
            "used to quantitatively rank and group customers based on the recency, frequency and monetary total of their recent transactions.")


col1, col2, col3 = st.columns([3,6,1])

with col1:
    st.write("")

with col2:
    st.image("DS-620-Data_Visualization/DVProject.png", caption="Project Section Allocation", width= 700)

with col3:
    st.write("")

st.markdown("From the figure above, the group members in bold where taking 'point' on the respective section of the project"
            " meaning they where the ones in charge of that particular part of the project. "
            "So Supriya was in charge of the data collection and data preparation with the help of Kuzi and Sahil. "
            "Manideep, with the assistance of Kuzi, Sahil and Supriya was in charge the EDA part of the project, "
            "making preliminary data visualization in Tableau and then translating the visualization in python using Plotly. "
            "Sahil had point on the Model building and evaluation, with assistance from Kuzi. And lastly, Kuzi, with the assistance "
            "of Supriya, worked on the Model Deployment/Presentation that is creating the Streamlit Application and the Medium Page")

matplotlib.use("agg")

_lock = RendererAgg.lock

@st.cache
def data_loading(path):
    """
    This function reads the Excel data file from UCI Machinelearning repository 
    The data is stored into a pandas dataframe that the fuction returns  
    """
    df = pd.read_csv(path)
    return df

data_path = "DS-620-Data_Visualization/Clean_Online_Retail_data.csv"

df = data_loading(data_path)

st.markdown("### **Data cleaning and preprocessing**")
st.markdown("**General Information About the Data**")
st.markdown("The dataset, [Online Retail](https://archive-beta.ics.uci.edu/ml/datasets/online+retail), was collected from the [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu), "
            "which is a dataset repository that currently has or maintains 588 data sets as a service to the machine learning community."
            "This is a transnational data set which contains all the transactions "
            "occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered "
            "non-store online retail. The company mainly sells unique all-occasion gifts. "
            "Many customers of the company are wholesalers.")

st.markdown("**Information about the Attributes/Columns in the Dataset**")
"""
* ***InvoiceNo:*** Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
* ***StockCode:*** Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
* ***Description:*** Product (item) name. Nominal.
* ***Quantity:*** The quantities of each product (item) per transaction. Numeric.
* ***InvoiceDate:*** Invice Date and time. Numeric, the day and time when each transaction was generated.
* ***UnitPrice:*** Unit price. Numeric, Product price per unit in sterling.
* ***CustomerID:*** Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
* ***Country:*** Country name. Nominal, the name of the country where each customer resides.
"""

col1, col2, col3 = st.columns([5,1,3])

with col1:
    st.image("DS-620-Data_Visualization/df_head().png", caption="Data before cleaning and prep", width= 1100)

with col2:
    st.write("")
with col3:
    st.image("DS-620-Data_Visualization/df_info().png", caption=" Data infor on the column data type and null values", width= 550)

st.markdown("From the above we can see that the Description and CustomerID columns have null values, "
            "and that the CustomerID has data type float64. ")

"""
* We are going to convert the CustomerID datatype to 'string' and ***Consider*** the null values in this columns ***Guest Customer*** implying that they are customers that do not have Customer IDs.
* For the null values in the Description column, we are also going to delete the rows, since we do not have any use of them in this particular project.
"""

st.markdown("**What we did to the data:**")

"""
* Changing the CustomerID to dtype 'object' and replace the null values to 'Guest Customer'.
* Remove the rows that have null values from the Description column. we are also going to remove rows that have product description that has length less than or equal to 6 characters, this is because the description are not actually products (well from our understanding). Below is like a sample of some of these descriptions 
> > ['wet?','Damaged','Missing','Discount','lost','MIA','SAMPLES','Display','mailout ','?','missing?','broken','lost??','CARRIAGE','mouldy','smashed']
* Cancelled orders will be removed, since as per the info about the attributes/columns of our dataset, for the 'InvoiceNo' the entries that contain a 'c' is a cancelled order.<br><br>
* We are going to remove the data entries from 2010 and then split the 'Invoice Date' column into just Date (without the time),Time, Month, Day, Week, Day of the Week, Hour, Time of Day (which is either Morning, Afternoon or Evening).We are going to need these to dissect the transations at different periods.
* A 'Sales Revenue' column will be created by multiplying the 'Quantity' and 'UnitPrice' columns. This will give the monetery value of each data entry.
"""

st.markdown("***The below is the first 5 rows of the cleaed dataset***")
st.dataframe(df.head())

st.markdown("### ***Exploratory Data Analysis (EDA)***")

"""
* Exploratory data analysis is an approach/practice of analyzing data sets to summarize their main characteristics, often using statistical graphics and other ***data visualization***. It is a critical process of performing initial ***investigations to discover*** patterns, detect outliers and anomalies, and to gain some new, hidden, insight into the data.
* Investigating questions like what the total volume of purchases per month, week, day of the week, time of the day right to the hour. We will look at customers more, later when we get into the ***Recency, Frequency and Monetary Analysis (RFM)*** sections of the project.
"""
"""
We are going to create a helper fuction that will group the Quantity and Sales Revenue with respect to either the 'CustomerID', 'Country', 'Date', 'Month','Week of the Year', 'Day of Week', 'Hour', or 'Time of Day'.
"""
@st.cache
def group_Quantity_and_SalesRevenue(df,string):
    """ 
    This function inputs the main data frame and feature name 
    The feature name is the column name that you want to group the Quantity and Sales Revenue
    """

    df = df[[f'{string}','Quantity','Sales Revenue']].groupby([f'{string}']).sum().sort_values(by= 'Sales Revenue', ascending = False).reset_index()

    return df


col1, col2, col3 = st.columns([3,3,3])
with col1:
    Country_Data1 = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)
    Country_Data2 = df[df['Country'] != "United Kingdom"].groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("With the UK", "Without the UK")
                        )
    fig.add_trace(go.Bar(x=Country_Data1['Country'], y=Country_Data1['InvoiceNo'],name = 'with the UK'),1, 1)

    fig.add_trace(go.Bar(x=Country_Data2['Country'], y=Country_Data2['InvoiceNo'],name = 'without the UK'),1, 2)

    fig.update_layout(showlegend=False, title_text="Top 10 Number of orders per Country")
    st.plotly_chart(fig)

    st.markdown("The above charts show that the UK by far has more invoices, "
                "just as suspected, with invoices surpassing 15K. Germany in "
                "second place, with approximately 30 time less invoices. The "
                "retail store management can start possing question of why this is the "
                "case, especially when this is a Online retail store.")


temp_df = group_Quantity_and_SalesRevenue(df,'Country')
X = temp_df[temp_df['Country'] != "United Kingdom"]
with col2:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("With the UK", "Without the UK")
                        )
    fig.add_trace(go.Bar(x=temp_df['Country'], y=temp_df['Quantity'],name = 'with the UK'),1, 1)

    fig.add_trace(go.Bar(x=X['Country'], y=X['Quantity'],name = 'without the UK'),1, 2)

    fig.update_layout(showlegend=False, title_text="Quantity of orders per Country")
    st.plotly_chart(fig)

    st.markdown("The next 4 charts are looking at how the countries fare up with regards to the "
                "***Quantity sold*** and ***Sales Revenue***.The first 2 above are "
                "for the Quantity sold and the last 2 for Sales "
                "Revenue both for the whole year of 2011.")

with col3:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("With the UK", "Without the UK")
                        )
    fig.add_trace(go.Bar(x=temp_df['Country'], y=temp_df['Sales Revenue'],name = 'with the UK'),1, 1)

    fig.add_trace(go.Bar(x=X['Country'], y=X['Sales Revenue'],name = 'without the UK'),1, 2)

    fig.update_layout(showlegend=False, title_text="Sales Revenue of orders per Country")
    st.plotly_chart(fig)


    st.markdown("Just as expected, the UK has high volumes of Quantitly "
                "sold and the below charts should show that the UK has high "
                "sales as well. However, unlike the number of invoices, the "
                "Netherlands has the second highest volume of Quantity sold at "
                "approximately 200K.")

st.markdown("Question like, what "
            "is the web traffic like to the store web page, or should they start thinking of "
            "***Search Engine Optimization (SEO)***, which is the process of improving the "
            "quality and quantity of website traffic to a website or a web page from search engines. "
            "Many other questions can be raised from the charts above.")

st.markdown("We now going to look at the products, which ones have high Quantity "
            "sold, or which product has high Sales Revenue. But first the below "
            "chart is a wordcloud of the product descriptions. A Word Clouds is a visual "
            "representations of words that give greater prominence to words that appear "
            "more frequently, in this case the frequency is the 'Quantity'")


@st.cache
def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0\xa0\xa0"
    display(HTML(output))


row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 1, .1, 1, .1))
st.set_option('deprecation.showPyplotGlobalUse', False)
with row3_1, _lock:
    st.markdown("Word Cloud of the Product Descriptions with Quantity as the frequency")
    #we can create a 
    temp_df = pd.DataFrame(df.groupby('Description')['Quantity'].sum()).reset_index()
    title = "Product Description"
    plt.figure(figsize=(20,40)) 
    tuples = [tuple(x) for x in temp_df.values]
    wordcloud = WordCloud().generate_from_frequencies(dict(tuples))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    st.pyplot()

with row3_2, _lock:
    st.markdown("Word Cloud of the Product Descriptions with Sales Revenue as the frequency")
    #we can create a word cloud of the product descriptions
    temp_df = pd.DataFrame(df.groupby('Description')['Sales Revenue'].sum()).reset_index()
    title = "Product Description"
    plt.figure(figsize=(20,40)) 
    tuples = [tuple(x) for x in temp_df.values]
    wordcloud = WordCloud().generate_from_frequencies(dict(tuples))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title)
    plt.show()
    st.pyplot()
 
st.write('')

st.markdown("### **Monthly Stats**") 

st.markdown("The below are the monthly analysis of the Sales Revenue and the Quantity of iterms sold")
temp_df = group_Quantity_and_SalesRevenue(df,'Month')

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row4_1, _lock:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Quantity", "Sales Revenue")
                        )
    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Monthly Sales Revanue and Quantity")
    st.plotly_chart(fig)


    st.markdown("The above graphs show the monthly trend of Quantity of "
                "products ordered(left) and Sales Revenue(right). Both "
                "the measures were the highest "
                "in Novemeber folllowed by October and Septemeber.")

with row4_2, _lock:
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=("Quantity per Month", "Sales Revenue per Month")
                    )
    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Month'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Month'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(height=450, width=800, title_text="Percentage pie charts for Monthly Sales Revanue and Quantity")

    st.plotly_chart(fig)

    st.markdown("The above pie charts depicts the quantity of products ordered "
                "and sales revenue per month with highest in the month of November "
                "with 14.4%, and lowest in the month of february with 5.43%.")
  
st.write('')

st.markdown("### **Weekly Stats**")
st.markdown("The below are the weekly analysis of the Sales and the Quantity of iterms sold")

row5_1, row5_2, row5_3 = st.columns([3,6,1])

temp_df = group_Quantity_and_SalesRevenue(df,'Week of the Year')

with row5_1, _lock:
    st.write("")

with row5_2, _lock:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Weekly Sales Revanue and Quantity")
    st.plotly_chart(fig)

    st.markdown("The above graphs shows the weekly trend of sales revenue "
                "and the quantity of products ordered. The highest peak was on "
                "the 49th week in the month of November. As it's a holiday season, "
                "there was a high demand for the decoration items. As the quantity "
                "increases sales revenue too increases.")


with row5_3, _lock:
    st.write("")


st.markdown("### **Daily Stats**") 
st.markdown("The below are the daily analysis of the Sales and the Quantity of iterms sold")

row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns(
    (.1, 1, .1, 1, .1))
temp_df = group_Quantity_and_SalesRevenue(df,'Day of Week')

with row6_1, _lock:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Quantity", "Sales Revenue")
                        )
    fig.add_trace(go.Bar(x=temp_df['Day of Week'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Day of Week'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(coloraxis=dict(colorscale='Greys'), showlegend=False, title_text="Day of the Week Sales Revanue and Quantity")
    st.plotly_chart(fig)

    st.markdown("The above graphs depict the daily trend of Sales revenue and "
                "quantity. Thursday was observed to generate the highest "
                "quantity of products and Sales Revenue.")
        
with row6_2, _lock:
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=("Quantity", "Sales Revenue")
                    )
    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Day of Week'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Day of Week'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(height=400, width=800, title_text="Percentage pie charts for Day of the Week Sales Revanue and Quantity")

    st.plotly_chart(fig)

    st.markdown("The above pie charts shows the day of the week"
                " percentage trend of sales revenue and quantity "
                "of products ordered.")

st.write('')

st.write('')
row8_space1, row8_1, row8_space2, row8_2, row8_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row8_1, _lock:
    temp_df = group_Quantity_and_SalesRevenue(df,'Time of Day')
    fig = make_subplots(rows=1, cols=2,
                    specs=[[{"type": "pie"}, {"type": "pie"}]], 
                    subplot_titles=("Quantity", "Sales Revenue")
                    )
    fig.add_trace(
        go.Pie(values = temp_df['Quantity'], labels = temp_df['Time of Day'],
        name = 'Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Sales Revenue'], labels = temp_df['Time of Day'],
        name = 'Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(height=400, width=800, title_text="Percentage pie charts for Time of Day Sales Revanue and Quantity")

    st.plotly_chart(fig)

    st.markdown("The above piecharts shows the breakdown of Quantity of orders(left) "
                "and Sales revenue(right) by time of the day.  More than 99%, of "
                "the orders were placed during mornings and afternoon.")

with row8_2, _lock:
    temp_df = group_Quantity_and_SalesRevenue(df,'Hour')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Quantity", "Sales Revenue")
                        )
    fig.add_trace(go.Bar(x=temp_df['Hour'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Hour'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(coloraxis=dict(colorscale='Greys'), showlegend=False, title_text="Hourly Sales Revenue and Quantity")
    st.plotly_chart(fig)

    st.markdown("The above graphs shows the hourly trend of sales revenue "
                "and quantity of products ordered. As expected, the hours between "
                "9 am - 3 pm were found to be the busiest times.")

st.write('')

temp_df = group_Quantity_and_SalesRevenue(df, 'Description')
Quantity_tempA = temp_df.sort_values(ascending=False, by = "Quantity").head(10).reset_index(drop=True)
Quantity_tempB = temp_df.sort_values(ascending=False, by = "Quantity").tail(10).reset_index(drop=True)
Quantity_tempA.drop('Sales Revenue', axis=1, inplace=True)
Quantity_tempB.drop('Sales Revenue', axis=1, inplace=True)

row9_space1, row9_1, row9_space2, row9_2, row9_space3 = st.columns(
    (.1, 1, .1, 1, .1))
with row9_1, _lock:
    st.subheader("Top 10 and Bottom 10 Product Description by Quantity")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Top 10 Product", "Bottom 10 Products")
                        )

    fig.add_trace(go.Bar(x=Quantity_tempA['Description'], y=Quantity_tempA['Quantity'],name = 'Top10'),1, 1)

    fig.add_trace(go.Bar(x=Quantity_tempB['Description'], y=Quantity_tempB['Quantity'],name = 'Bottom10'),1, 2)

    fig.update_layout(showlegend=False, title_text="Product Description by Volume Quantity")
    st.plotly_chart(fig)

    st.markdown("The above graphs depict the top 10 and bottom 10 products by "
                "volume quantity. The product paper craft, little birdie was the "
                "most popular item whereas orange/fuschia stones necklace was the "
                "least popular item.")

Sales_Revenue_tempA = temp_df.sort_values(ascending=False, by = "Sales Revenue").head(10).reset_index(drop=True)
Sales_Revenue_tempB = temp_df.sort_values(ascending=False, by = "Sales Revenue").tail(10).reset_index(drop=True)
Sales_Revenue_tempA.drop('Quantity', axis=1, inplace=True)
Sales_Revenue_tempB.drop('Quantity', axis=1, inplace=True)

with row9_2, _lock:
    st.subheader("Top 10 and Bottom 10 Product Description by Sales Revenue")
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                  subplot_titles=("Top 10 Product", "Bottom 10 Products")
                        )

    fig.add_trace(go.Bar(x=Sales_Revenue_tempA['Description'], y=Sales_Revenue_tempA['Sales Revenue'],name = 'Top10'),1, 1)

    fig.add_trace(go.Bar(x=Sales_Revenue_tempB['Description'], y=Sales_Revenue_tempB['Sales Revenue'],name = 'Bottom10'),1, 2)

    fig.update_layout(showlegend=False, title_text="Product Description by Sales Revenue")
    st.plotly_chart(fig)

    st.markdown("The above graphs depict the top 10 and bottom 10 products by sales "
                "revenue. The product paper craft, little birdie was found to be "
                "generating the highest sales and pads had the lowest sales.")

st.write('')

st.markdown("### **Recency, Frequency, Monetary Value (RFM) Analysis**")

st.markdown("What is RFM Analysis and how does this type of analysis benefit "
            "a business, in this case, this Online Retail Business?")
st.markdown("As defined at the top of this page, RFM Analysis is an analytical "
            "technique used to quantitatively rank and group customers based on "
            "the recency, frequency, and monetary value from a business’ "
            "transactional data.")
"""
* Recency is how recently a customer has made a purchase to the store 
* Frequency is how often a customer makes a purchase
* Monetary Value is how much money a customer spends on purchases
"""
st.markdown("The analysis evaluates customers by scoring them in three "
            "categories on how recently they've made a purchase, how often "
            "they buy, and the monetary value of their purchases. This in turn "
            "helps the business somewhat predict which customers are their "
            "regulars, occasional customers, and new customers, how much revenue "
            "comes from these respective groups of customers, and which type of "
            "customer they need to target in advertainment and promotional "
            "campaigns. This RFM Analysis, together with Market Basket Analysis "
            "aids business management to make data-driven decisions in Inventory "
            "management, product marketing promotion, customer analysis, and "
            "retention. These two techniques are intertwined, as for example, the "
            "frequency of a customer’s transactions may be affected by the type "
            "of products stocked (and availability)  in the store, and the price "
            "of the products.")

st.markdown("**The first** thing that we are going to need is the reference "
            "date in this case the day after the last recorded date in the "
            "dataset plus a day")
            
ref_date = datetime.strptime('2011-12-10 12:50:00', '%Y-%m-%d %H:%M:%S')

st.markdown('So, our **reference date** is 2011-12-10 12:50:00')

col1, col2, col3 = st.columns([5,0.1,5])

with col1:
    data = df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo', title='Graph of top ten customer with respect to the invoice number')
    st.plotly_chart(fig)

    st.markdown("")


with col3:
    temp_df = df[df["CustomerID"] != "Guest Customer"]
    data = temp_df.groupby("CustomerID")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo', title='Graph of top ten customer with respect to the invoice number without the Guest Cusstomer')
    st.plotly_chart(fig)


    st.markdown("")


st.markdown("For RFM we are going to remove the 'Guest Customer', the customers"
            " without Customer ID")

df_temp = df[df['CustomerID'] != "Guest Customer"]

RFM_df = df_temp.groupby('CustomerID').agg({'InvoiceDate': lambda x: (ref_date - datetime.strptime(x.max(), '%Y-%m-%d %H:%M:%S')).days,
                                    'InvoiceNo': lambda x: x.nunique(),
                                    'Sales Revenue': lambda x: x.sum()})

RFM_df.columns = ['Recency', 'Frequency', 'Monetary']
RFM_df["R"] = pd.qcut(RFM_df['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
RFM_df["F"] = pd.qcut(RFM_df['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
RFM_df["M"] = pd.qcut(RFM_df['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
RFM_df['RFM_Score'] = (RFM_df['R'].astype(int)+RFM_df['F'].astype(int)+RFM_df['M'].astype(int))

RFM_df.reset_index(inplace=True)

#st.table(RFM_df.head(10))

temp = RFM_df[["RFM_Score", "Recency", "Frequency", "Monetary"]].groupby("RFM_Score").agg(["mean"])

#st.table(temp)
col1, col2, col3 = st.columns([5,0.1,5])

with col1:
    st.markdown("**Table of the RFM metrics**")
    st.table(RFM_df.head(10))
    st.markdown("This is the RFM table, which contains the Recency, Frequency, "
                "and the Monetary value per customer ID. "
                "The table also has the R,F,M values and the RFM Score"
                "The RFM Score is a summation of the 3 R+F+M values, therefor it "
                "ranges from 3 to 12 with 3 being the customer that last visited the "
                "store a long time ago, not a frequent visitor and spends little money")


with col3:
    st.markdown("**Table of Recency, Frequency Monetary value means group by RFM Score**")
    st.table(temp)
    st.markdown("We grouped the RFM scores with the mean values of recency, frequency, and"
                " monetary per each score. The table shows that customers with the lowest "
                "RFM scores have the highest recency value and the lowest frequency and "
                "monetary value and the opposie is true as well, customers with relative "
                "low recency have high monetary value")

st.markdown("We can call it good and be done with this, as from the above we gouped "
            "the RFM metric with respect to RFM Score, thus we have 9 groups of "
            "customer. But we are going to take a step further and actually do a "
            "cluster analysis using 'kmeans clustering' to see how best we can group "
            "our customers")

st.markdown("#### **K means Clastering**")

st.markdown("**We are going to do Kmeans Clustering for our Customers** using the "
            "RFM metrics table/dataframe that way we see how best our customers "
            "can be group.")
"""
We will use the **Elbow Graph** to find the best value of the clusters
"""

#now we will use the standard scalr function from SKlean preprocessing and scale transform fit the data (without the Customer Id column)
scaler = StandardScaler()
RFM_df_log = RFM_df[['Recency','Frequency','Monetary','RFM_Score']]
RFM_df_scaled = scaler.fit_transform(RFM_df_log)
RFM_df_scaled = pd.DataFrame(RFM_df_scaled)
RFM_df_scaled.columns = ['Recency','Frequency','Monetary','RFM_Score']

#st.table(RFM_df_scaled.head())

the_scores = []
for i in range(2,11): #we will explore from 2 to 10 clusters
  kmeans = KMeans(n_clusters=i, init='k-means++',n_init=10,max_iter=50,verbose=0)
  kmeans.fit(RFM_df_scaled)
  the_scores.append(kmeans.inertia_)

col1, col2, col3 = st.columns([.1,.1,.1])

with col1:
    st.write("")

with col2:
    st.subheader("Lets plot the Elbow")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax = plt.plot(range(2,11,1),the_scores)
    ax = plt.title('The Elbow Method Graph')
    ax = plt.xlabel('Number of clusters')
    ax = plt.xticks(ticks=range(2,11))
    st.pyplot(fig)

with col3:
    st.write("")

st.markdown("From the Elbow graph, it looks like we should be using 4 "
                "clusters, but to verfy, we will calculate the ‘silhouette_score’ "
                "for our clusters. Silhouette Coefficient or silhouette score is "
                "a metric used to calculate the goodness of a clustering technique. "
                "Its value ranges from -1 to 1.1 means clusters are well apart from "
                "each other and clearly distinguished.0 means clusters are "
                "indifferent, or we can say that the distance between clusters is "
                "not significant and -1 means clusters are assigned in the wrong way.")
    
st.markdown("Calculated the silhouette_score from the sklearn metrics module function silhouette_score")


output = st.empty()
with st_capture(output.code):
    for i in range(2,11): 
        kmeans = KMeans(n_clusters=i, init='k-means++',n_init=10,max_iter=50,verbose=0)
        kmeans.fit(RFM_df_scaled)
        silhouette = silhouette_score(RFM_df_scaled,kmeans.labels_)
        print(f"for {i} clusters the silhouette_score is {silhouette}")
        sleep(1)

st.markdown("From the above and using best judgment, we are going to use 4 clusters. "
            " Below are the box plot of the 4 clusters with respect to the Monetary "
            "value, Frequency and Recency.")

kmeans = KMeans(n_clusters=4, init='k-means++',n_init=10,max_iter=50,verbose=0)
kmeans.fit(RFM_df_scaled)

RFM_df['Clusters'] = kmeans.labels_

#st.table(RFM_df.head())

row10_space1, row10_1, row10_space2, row10_2, row10_space3, row10_3,row10_space4 = st.columns(
    (.1, 2, .1, 2, .1,2,.1))


with row10_1, _lock:
    sns.boxplot( x= 'Clusters',y= 'Monetary' ,data=RFM_df)
    plt.title("BoxPlot of Clusters v Monetary")
    plt.show()
    st.pyplot()

with row10_2, _lock:
    sns.boxplot( x= 'Clusters',y= 'Frequency' ,data=RFM_df)
    plt.title("BoxPlot of Clusters v Frequency")
    plt.show()
    st.pyplot()

with row10_3, _lock:
    sns.boxplot( x= 'Clusters',y= 'Recency' ,data=RFM_df)
    plt.title("BoxPlot of Clusters v Recency")
    plt.show()
    st.pyplot()

st.write('')

temp_df = RFM_df[["Clusters","RFM_Score", "Recency", "Frequency", "Monetary"]].groupby("Clusters").agg(["mean"])
temp_df.columns = ["RFM_Score mean", "Recency mean", "Frequency mean", "Monetary mean"]


col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.subheader("This a table of the resulting 4 clusters from the RFM "
            "analysis and Kmeans Clustering")
    st.table(temp_df)
    st.markdown("The mean RFM Score, Recency, Frequency, and  Monetary "
                "value with respect to the clusters. the clusters with high "
                "RFM Score have high mean monetary value and relative high frequency "
                "mean.")
    st.write('')

with col3:

    st.write("")



col4, col5, col6 = st.columns([3,6,1])

with col4:
    st.write("")

with col5:
    st.subheader("The below Pie Charts are the Results of the RFM Analysis")
    specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("RFM_Score", "Recency", "Frequency","Monetary"),
                        specs=specs
                        )

    fig.add_trace(
        go.Pie(values = temp_df['RFM_Score mean'], labels = temp_df.index,
        name = 'RFM_Score'),
        1, 1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Recency mean'], labels = temp_df.index,
        name = 'Recency'),
        1, 2
    )
    fig.add_trace(
        go.Pie(values = temp_df['Frequency mean'], labels = temp_df.index,
        name = 'Frequency'),
        2, 1
    )
    fig.add_trace(
        go.Pie(values = temp_df['Monetary mean'], labels = temp_df.index,
        name = 'Monetary'),
        2, 2
    )
    fig.update_layout(height=800, width=800, title_text=" ")
    st.plotly_chart(fig)

    st.write("")




with col6:
    st.write("")

st.markdown("The above pie charts show the percent representation by each cluster "
            "for the RFM metrics. The customers in the 'blue cluster' have a high "
            "average monetary value and high average frequency. The recency of these "
            "customers is very low among the 4 clusters with 2.3%. The 'red cluster' "
            "contains customers with a very low monetary value of 3.1%, recency of 6.52%, "
            "and a frequency of 10.1%, this by far is the customers with poor metrics; "
            "however, they are better than 'purple' and 'green' customer as far as monetary "
            "value is concerned. These red customers, at a glance, are probably customers "
            "that the store has had for a long period, but starting to lose interest in the "
            "store, these customers can be prime candidates for targeted promotions. The "
            "management must drill down on this group of customers looking at what they "
            "purchase and finding the best ways to reel them back. They can even do a "
            "filtered Market Basket Analysis just for these customers. The 'purple' "
            "customers have high recency and low monetary value and frequency, these might "
            "be new customers, management can create targeted promotions for this group to "
            "entice them to purchase more and frequently. The green customers are somewhat "
            "similar to 'red' customers, but with a very less monetary value of 0.566%. A "
            "filtered MBA can help uncover the buying habits of these customers, which in "
            "turn can bring up some ideas on how to improve the monetary and frequency of "
            "this group.")

st.markdown("### **Market Basket Analysis (MBA)**")

st.markdown("Market basket analysis is a process that looks for relationships or "
            "associations of objects that “go together”, it is the analysis of an "
            "assortment of items to identify affinities between the items. MBA aims"
            " to find relationships and establish patterns, and in the case of this "
            "project, affinities between products with respect to Invoice IDs in the "
            "Online Retail transaction data. ")

st.markdown("Going with the business proposition of this project, market basket "
            "analysis is one of the most important tools used to aid business "
            "management make data-driven decisions. Decisions such as product "
            "placement, for example, identifying products that usually are "
            "co-purchased together and placing the products in close proximity in a "
            "way that encourages the customer to buy or “not forget” to purchase. Market "
            "basket analysis in a retail setting, apart from being useful in crossing "
            "selling products, can be very useful in customer retention. ")

st.markdown("From the RFM analysis that we conducted earlier, together with the "
            "use of the ***K-means clustering***, customers were clustered into 4 groups "
            "with different customer recency, frequency, and monetary value metrics, "
            "MBA can then be used to dissect further the different customer purchasing "
            "habits. It can be used to determine the right promotions, incentives, "
            "product discounts that can be offered in order to retain the customer’s "
            "business. ")

st.markdown("MBA uncovers these meaningful correlations between different entities "
            "according to their co-occurrence in a data set by generating "
            "association rules. Association rule learning is a machine learning "
            "method for discovering relationships and establishing patterns between "
            "variables in a dataset or collection of items. ***Support***, ***Confidence***, and ***Lift***"
            " are three important evaluation criteria of association discovery")

"""
Support value is computed as the joint probability (relative frequency of cooccurrences) of the body and the head of each association rule. This is expressed by the quantity
"""
st.image("DS-620-Data_Visualization/support.png", width= 340)

"""
Confidence value denotes the conditional probability of the head of the association rule, given the body of the association rule, expressed as
"""
st.image("DS-620-Data_Visualization/confidence.png", width= 350)

"""
Lift value measures the confidence of a rule and the expected confidence that the second product will be purchased depending on the purchase of the first product expressed as
"""
st.image("DS-620-Data_Visualization/lift.png", width= 350)

st.markdown("")

#for the UK we will call this the UKbasket
st.markdown("For the Market basket analysis we filtered the data to only include "
            "United Kingdom. Since this was our first time creating a streamlit app"
            " we thought it was best to keep thing relatively simple.")

country = "United Kingdom"


basket = (df[df['Country'] == country]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#from the above table we are going to make that very value that is less than or equal to 0 be zero and anything else be equal to 1.
#this is called one hot encoding and we will use the buit in pandas applymap() function with a custom function as input to check the condition stated above
def encoder(num):
    if num <= 0: return 0
    if num >= 1: return 1

#now we encode
basket = basket.applymap(encoder)

frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

#st.table(frequent_itemsets.head(10))

#we then can look at the rules  of association using the MLxtend association_rules() 
#the function generates a DataFrame of association rules including the metrics 'confidence', and 'lift'
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

import re
@st.cache
def search(x):
    text = str(x)
    m = re.search("{(.+?)}", text)
    found = m.group(1)
    return found


@st.cache
def che(x):
    if x <= tempo["25%"]:
        return "<=25%"
    elif x > tempo["25%"] and x <= tempo["50%"]:
        return "<=50%"
    elif x > tempo["50%"] and x <= tempo["75%"]:
        return "<=75%"
    else:
        return ">75%"


st.write("") 

st.markdown("The below visualizations are the result of the market basket "
            "analysis, which was done using the ***Apriori Algorithm***  from the "
            "python ***‘mlxtend’*** module ***Apriori algorithm*** is a commonly used algorithm "
            "that identifies itemsets that occur with support greater than a pre-defined"
            " value (frequency) and calculates the confidence of all possible rules based "
            "on those itemsets")

row14_space1, row14_1, row14_space2, row14_2, row14_space3 = st.columns(
    (.1, 1, .1, 1, .1))
with row14_1, _lock:
    st.markdown("Here you are able to filter the below Scatter plot using minimum confidence")
    confidence__min_max  = st.slider("Select the minimum Confidence", 0.0850,0.800)
    st.write("")

with row14_2, _lock:
    st.markdown("Here you are able to filter the below Scatter plot using minimum support")
    support_min_max  = st.slider("Select the minimum Support", 0.0102, 0.040)
    st.write("")

 
st.write('')


rules = rules[rules['support'] >= 0.02]
rules = rules[rules['confidence'] >= 0.41]

rules['itemsets'] = rules["antecedents"].apply(lambda x: search(x)) +" => "+rules["consequents"].apply(lambda x: search(x))
tempo = dict(rules.lift.describe())
rules["lift checker"] = rules["lift"].apply(lambda x: che(x))

st.markdown("### **Top Confidence and Support itemsets**")

row12_space1, row12_1, row12_space2, row12_2, row12_space3 = st.columns(
    (.1, 1, .1, 1, .1))
with row12_1, _lock:
    fig = px.scatter(rules.sort_values(by="confidence",ascending= True).head(50), 
                    x="support", 
                    y="confidence",
                    size="lift",
                    color="lift checker",
                    title="Scatter Plot of Confidence vs Support, with size = lift and color range of lift (fltr = conf)", 
                    hover_name="itemsets", 
                    log_x=True, size_max=60)
    st.plotly_chart(fig)


with row12_2, _lock:
    fig = px.scatter(rules.sort_values(by="support",ascending= True).head(50), 
                    x="support", 
                    y="confidence",
                    size="lift",
                    color="lift checker",
                    title="Scatter Plot of Confidence vs Support, with size = lift and color range of lift (fltr = sup)", 
                    hover_name="itemsets", 
                    log_x=True, size_max=60)
    st.plotly_chart(fig)

 
st.write('')

st.markdown("### **Bottom Confidence and Support itemsets**")

row13_space1, row13_1, row13_space2, row13_2, row12_space3 = st.columns(
    (.1, 1, .1, 1, .1))
with row13_1, _lock:
    fig = px.scatter(rules.sort_values(by="confidence",ascending= True).tail(50), 
                    x="support", 
                    y="confidence",
                    size="lift",
                    color="lift checker",
                    title="Scatter Plot of Confidence vs Support, with size = lift and color range of lift (fltr = conf)", 
                    hover_name="itemsets", 
                    log_x=True, size_max=60)
    st.plotly_chart(fig)



with row13_2, _lock:
    fig = px.scatter(rules.sort_values(by="support",ascending= True).tail(50), 
                    x="support", 
                    y="confidence",
                    size="lift",
                    color="lift checker",
                    title="Scatter Plot of Confidence vs Support, with size = lift and color range of lift (fltr = sup)", 
                    hover_name="itemsets", 
                    log_x=True, size_max=60)
    st.plotly_chart(fig)
 
st.write('')



st.markdown("### ***Conclusion***")

st.markdown("This was a very interesting project which covered a wide array"
            " of topics that seem different but quite similar. For further "
            "research or analysis, it would be interesting to do the market "
            "basket analysis on the clusters that resulted (with the help of the"
            " K-means clustering) from the recency, frequency, and monetary value "
            "analysis and do market basket analysis on each of them")
