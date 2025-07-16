from operator import sub
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.title("Sentiment Analysis App for Tweets")
st.sidebar.title("Sentiment Analysis App")
st.markdown("This app performs sentiment analysis on text data from tweets 🐦.")
st.sidebar.markdown("This app performs sentiment analysis on text data from tweets 🐦.")

@st.cache_data(persist=True)
def load_data():
    data=pd.read_csv("Tweets.csv")
    data['tweet_created']=pd.to_datetime(data['tweet_created'])
    return data

data = load_data()
#st.write(data)

st.sidebar.subheader("Show random tweet")
rt=st.sidebar.radio("Sentiment",("positive","negative","neutral"))

filtered_data = data.query('airline_sentiment == @rt')[["text"]]
if not filtered_data.empty:
    st.sidebar.markdown(filtered_data.sample(n=1).iat[0, 0])
else:
    st.sidebar.markdown("No samples available for selected sentiment.")

st.sidebar.markdown("### Number of tweets by sentiment")
selected= st.sidebar.selectbox("Visusalization Type",("Histogram","Pie Chart"),key="1")

sentiment_counts = data['airline_sentiment'].value_counts()
sentiment_counts = pd.DataFrame({
    'Sentiment': sentiment_counts.index,
    'Count': sentiment_counts.values
})

if not st.sidebar.checkbox("Hide",True):
    if selected == "Histogram":
        fig = px.histogram(sentiment_counts, x='Sentiment', y='Count', color='Sentiment', title="Number of Tweets by Sentiment")
        st.plotly_chart(fig)
    else:
        fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Number of Tweets by Sentiment")
        st.plotly_chart(fig)
               
st.sidebar.subheader("When and Where are people tweeting from?")
hour = st.sidebar.slider("Hour of Day", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key="2"):
    st.markdown("Showing data at {}:00 with {} tweets".format(hour, modified_data.shape[0]))
    st.map(modified_data)
    
if not st.sidebar.checkbox("Show Raw Data", False):
    st.subheader("Raw Data")
    st.write(data)
    
st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice = st.sidebar.multiselect("Pick Airlines", ("American","United","Southwest","US Airways","Virgin America","Delta"), key="3")
if len(choice) > 0:
    choice_data = data[data['airline'].isin(choice)]
    fig = px.histogram(choice_data, x='airline', y='airline_sentiment',histfunc='count',color='airline_sentiment',facet_col='airline_sentiment',labels={'airline_sentiment':'Sentiment'},height=600,width=800,title="Sentiment Analysis of Selected Airlines")
    st.plotly_chart(fig)
    
st.sidebar.subheader("Word Cloud")
word_sentiment = st.sidebar.radio("Word Cloud Sentiment", ("positive", "negative", "neutral"), key="4")
if not st.sidebar.checkbox("Close", True, key="5"):
    st.subheader("Word Cloud for {}".format(word_sentiment))
    word_data = data[data['airline_sentiment'] == word_sentiment]
    word_data = ' '.join(word_data['text'])
    processed_word_data = ' '.join([word for word in word_data.split() if 'http' not in word and not word.startswith('@') and word!= 'RT'])
    wordcloud = WordCloud(width=800, height=640, background_color='white', stopwords=STOPWORDS).generate(processed_word_data)
    plt.imshow(wordcloud.to_array(), interpolation='bilinear')
    plt.axis('off') 
    plt.show()
    plt.xticks([])
    plt.yticks([])  
    st.pyplot(plt)
