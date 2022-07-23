from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
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

@st.cache(allow_output_mutation=True, ttl= 600)
def load_data():
    """
    This fuction loads data from the aws rds mysql table
    """
    data = None
    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online_Retail_Data"
        data = pd.read_sql(query,engine)

    except Exception as e:
        print(str(e))
    
    return data
#loading the data
df = load_data() 



st.markdown("#### ***Lets take a look at the data:***")
"""
We are going to use the pandas `.shape` function/method to the total number of columns and rows of the dataframe. We can see that our dataframe contains 481313 rows and 16 columns

We'll use the pandas `.info()` function so see the general infomation (data types, null value count, etc.) about the data.
"""
st.markdown(f"###### ***The shape of the data***: {df.shape}")


col1, col2,col3 = st.columns((1, 0.01,.5))

df_head = pd.read_csv("df_head.csv")
with col1:
    st.markdown("***The below is the first 5 rows of the cleaed dataset***")
    st.dataframe(df_head)
with col2:
    pass
df_info = pd.read_csv("df_info.csv", index_col=0)
with col3:
    st.markdown("***The below is the info of the data***")
    st.dataframe(df_info)

st.success("If you want to take a look at how the data was cleaned, you "
            "can go check out the jupyter notebook of this project at: "
            "https://github.com/kkrusere/Market-Basket-Analysis-on-the-Online-Retail-Data/blob/main/MBA_Online-Retail_Data.ipynb")

######################functions############################

@st.cache(allow_output_mutation=True)
def group_Quantity_and_SalesRevenue(df,string):
    """ 
    This function inputs the main data frame and feature name 
    The feature name is the column name that you want to group the Quantity and Sales Revenue
    """

    df = df[[f'{string}','Quantity','Sales Revenue']].groupby([f'{string}']).sum().sort_values(by= 'Sales Revenue', ascending = False).reset_index()

    return df

#@st.cache(allow_output_mutation=True)
def wordcloud_of_Description(df, title):
    """
    This fuction creates a word cloud
    inputs a data frame converts it to tuples and uses the input 'title' as the title of the word cloud
    """
    plt.rcParams["figure.figsize"] = (20,20)
    tuples = [tuple(x) for x in df.values]
    wordcloud = WordCloud(max_font_size=100,  background_color="white").generate_from_frequencies(dict(tuples))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize = 27)
    plt.show()


country_list = ["All"] + list(dict(df['Country'].value_counts()).keys())
@st.cache(allow_output_mutation=True)
def choose_country(country = "all", data = df):
  """
  This fuction takes in a country name and filters the data frame for just country
  if the there is no country inputed the fuction return the un filtered dataframe
  """
  if country == "all":
    return data
  else:
    temp_df = data[data["Country"] == country]
    temp_df.reset_index(drop= True, inplace= True)

    return temp_df
##################################################################################
st.markdown("---")
st.markdown(" <h3 style='text-align: center;'>Exploratory Data Analysis <i>(EDA)</i>:</h3>", unsafe_allow_html=True)
col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    * Exploratory data analysis is an approach/practice of analyzing data sets to summarize their main characteristics, often using statistical graphics and other ***data visualization***. It is a critical process of performing initial ***investigations to discover*** patterns, detect outliers and anomalies, and gain some new, hidden, insights into the data.
    * Investigating questions like the total volume of purchases per month, week, day of the week, time of the day right to the hour. We will look at customers more later when we get into the ***Recency, Frequency and Monetary Analysis (RFM)*** in the Customer Segmentation section of the project.
    """
with col3:
    pass
############################################
col1, col2, col3= st.columns((1,.1,1))

with col1:
    Country_Data = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)
    fig = px.bar(Country_Data, x= "InvoiceNo", y='Country', title= "Top 10 Number of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    st.markdown("UK has more number of orders witk 16k Invoice numbers")

with col2:
    pass

with col3:
    Country_Data = df[df['Country'] != "United Kingdom"].groupby("Country")["InvoiceNo"].nunique().sort_values(ascending = False).reset_index().head(10)
    fig = px.bar(Country_Data, x= "InvoiceNo", y='Country', title= "Top 10 Number of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
########################################################

col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    The above charts show that the UK by far has more invoices, just as suspected, with invoices surpassing 15K. Germany in in second place, with approximately 30 time less invoices. The retail store management can start possing question of why this is the case, especially when this is a Online retail store. Question like, what is the traffic like to the store web page, or should they start thinking of ***Search Engine Optimization (SEO)***, which is the process of improving the quality and quantity of website traffic to a website or a web page from search engines. Many other questions can be raised from the 2 charts above.

    Below, we can take a look at how the countries fare up with regards to the ***Quantity sold*** and ***Sales Revenue***.The first plot is going to be for Quantity sold and the second will be for Sales Revenue both for the whole year of 2011.
    """
with col3:
    pass

####################
col1, col2, col3= st.columns((1,.1,1))
with col1:
    #choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df, x= "Quantity", y='Country', title= "Quantity of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:

    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x= "Quantity", y='Country', title= "Quantity of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)


####################

col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    Just as expected, the UK has high volumes of Quantitly sold and the below charts should show that the UK has high sales as well. However, unlike the number of invoices, the Netherlands has the second highest volume of Quantity sold at approximately 200K. 
    """
with col3:
    pass

####################
col1, col2, col3= st.columns((1,.1,1))
with col1:
    #choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df, x= "Sales Revenue", y='Country', title= "Sales Revenue of orders per country with the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:
    temp_df = group_Quantity_and_SalesRevenue(df,'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x= "Sales Revenue", y='Country', title= "Sales Revenue of orders per country without the UK")
    #fig.show(renderer='png', height=700, width=1000)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

####################
col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    The sales revenue of Netherlands and Germany is quite close. It would be interesting to see this broken down by time periods: 'Month', 'Week', 'Day of the Week', 'Time of Day' ,or 'Hour'.
    
    We now going to look at the products, which ones have high Quantity sold, or which product has high Sales Revenue. But first the below chart is a wordcloud of the product descriptions. A wordcloud is a visual representations of words that give greater prominence to words that appear more frequently, in this case the frequency is the 'Quantity'
    """
with col3:
    pass

####################
#here we ask the user to select a country to  look at
col1, col2, col3= st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country to Analyze',
        country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")


st.markdown("###### **We can create a word cloud of the Product Descriptions per Quantity & Product Descriptions per Sales Revenue**")

col1, col2, col3= st.columns((1,.1,1))
with col1:
    temp_df = pd.DataFrame(df.groupby('Description')['Quantity'].sum()).reset_index()
    title = "Product Description per Quantity"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

with col2:
    pass

with col3:
    temp_df = pd.DataFrame(df.groupby('Description')['Sales Revenue'].sum()).reset_index()
    title = "Product Description per Sales Revenue"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

###############################################

st.markdown("##### **Monthly Stats:**") 
"""
Below are the monthly analysis of the Sales and the Quantity of iterms sold
"""

col1, col2, col3= st.columns((1,.3,1))
with col1:
    temp_df = group_Quantity_and_SalesRevenue(df,'Month')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Quantity", "Sales Revenue")
                    )
    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Month'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Monthly Sales Revanue and Quantity")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above graphs show the monthly trend of Quantity of products ordered(left) and Sales Revenue(right). Both the measures were the highest in Novemeber folllowed by October and Septemeber.
    """

with col2:
    pass

with col3:
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
    fig.update_layout(title_text="Percentage pie charts for Monthly Sales Revanue and Quantity")

    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above pie charts depicts the quantity of products ordered and sales revenue per month with highest in the month of November with 14.4% and lowest in the month of february with 5.43%.
    """

##############################
st.markdown("##### **Weekly Stats:**")
"""
The below are the weekly analysis of the Sales and the Quantity of iterms sold
"""
ccol1, col2, col3= st.columns((.5,1,.5))
with col1:
    pass
with col2:
    #choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df,'Week of the Year')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                    subplot_titles=("Quantity", "Sales Revenue")
                            )

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Quantity'],name = 'Quantity'),1, 1)

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'], y=temp_df['Sales Revenue'],name = 'Sales Revenue'),1, 2)

    fig.update_layout(showlegend=False, title_text="Weekly Sales Revanue and Quantity")
    #fig.show(renderer='png', height=700, width=1200)
    #fig.show( height=700, width=1000)
    st.plotly_chart(fig)

    """
    The above graphs shows the weekly trend of sales revenue and the quantity of products ordered. The highest peak was on the 49th week in the month of November. As it's a holiday season, there was a high demand for the decoration items. As the quantity increases sales revenue too increases.
    """

with col3:
    pass



###############################





st.markdown("----")



#########################################################################
st.markdown(" <h3 style='text-align: center;'>Market Basket Analysis <i>(MBA)</i>:</h3>", unsafe_allow_html=True)
r"""
**What is Market Basket Analysis?:**

Market Basket Analysis (MBA) is a data mining technique that is mostly used in the Retail Industry to uncover customer purchasing patterns and product relationships. The techniques used in MBA identify the patterns, associations, and relationships (revealing product groupings and which products are likely to be purchased together) in in frequently purchased items by customers in large transaction datasets collected/registered at the point of sale. The results of the Market Basket Analysis can be used by retailers or marketers to design and develop marketing and operation strategies for a retail business or organization.<br>
Market basket analysis mainly utilize Association Rules {IF} -> {THEN}. However, MBA assigns Business outcomes and scenarios to the rules, for example,{IF X is bought} -> {THEN Y is also bought}, so X,Y could be sold together. <br>

Definition: **Association Rule**

Let $I$= \{$i_{1},i_{2},\ldots ,i_{n}$\} be an itemset.

Let $D$= \{$t_{1},t_{2},\ldots ,t_{m}$\} be a database of transactions $t$. Where each transaction $t$ is a nonempty itemset such that ${t \subseteq I}$

Each transaction in D has a unique transaction ID and contains a subset of the items in I.

A rule is defined as an implication of the form:
$X\Rightarrow Y$, where ${X,Y\subseteq I}$.

The rule ${X \Rightarrow Y}$ holds in the dataset of transactions $D$ with support $s$, where $s$ is the percentage of transactions in $D$ that contain ${X \cup Y}$ (that is the union of set $X$ and set $Y$, or, both $X$ and $Y$). This is taken as the probability, ${P(X \cup Y)}$. Rule ${X \Rightarrow Y}$ has confidence $c$ in the transaction set $D$, where $c$ is the percentage of transactions in $D$ containing $X$ that also contains $Y$. This is taken to be the conditional probability, like ${P(Y | X)}$. That is,

* support ${(X \Rightarrow Y)}$ = ${P(X \cup Y)}$

* confidence ${(X \Rightarrow Y)}$ = ${P(X|Y)}$

The lift of the rule ${(X \Rightarrow Y)}$  is the confidence of the rule divided by the expected confidence, assuming that the itemsets $X$ and $Y$ are independent of each other.The expected confidence is the confidence divided by the frequency of ${Y}$.

* lift ${(X \Rightarrow Y)}$ = ${ \frac {\mathrm {supp} (X\cap Y)}{\mathrm {supp} (X)\times \mathrm {supp} (Y)}}$


Lift value near 1 indicates ${X}$ and ${Y}$ almost often appear together as expected, greater than 1 means they appear together more than expected and less than 1 means they appear less than expected.Greater lift values indicate stronger association

"""

st.markdown("----")
st.markdown(" <h3 style='text-align: center;'>Customer Segmentation:</h3>", unsafe_allow_html=True)
"""
* RFM (recency, frequency, monetary) Analysis
"""

st.markdown("----")
st.markdown(" <h3 style='text-align: center;'>Product recomendation <i>(people who bought this also bought)</i>:</h3>", unsafe_allow_html=True)
col1, col2, col3= st.columns((.1,1,.1))
with col1:
    pass
with col2:
    """
    The product recommendation part of this project is going to make use of the Association Rules that where uncovered in the MBA section. Product recomentation is basically one of the advantages of Market Basket Analysis where you can recommend to customers products that are in the same itemsets as the customer's current products.
    """
with col3:
    pass