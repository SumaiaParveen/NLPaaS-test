import streamlit as st
import pandas as pd
import base64
import plotly.express as px

import numpy as np
import re
import nltk
from textblob import TextBlob
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def sent():

    #st.subheader("Unsupervised Sentiment Analysis")
    st.markdown("<h3 style='text-align: center;'>Unsupervised Sentiment Analysis</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    while True:
        try:

            df = pd.read_csv(uploaded_file)
            df1 = df.copy()
            st.subheader("First Five Rows of the Input DatFrame")
            st.write(df.head())
                    
            col_name = st.text_input("Enter the name of the column with text data.")
            col_name = str(col_name)

            try:

                df1 = df1[[col_name]]
                df1[col_name] = df1[col_name].replace('', np.nan)
                df1 = df1.dropna()
                st.subheader("Column with Text Data")
                st.write(df1)

                df_textblob = df1.copy()
                df_vader = df1.copy()

                type_task = st.radio("Select from below", ("TextBlob", "Vader"))  

                if type_task ==  "TextBlob": 

                            st.subheader("TextBlob")
                            
                            df_textblob.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
                            df_textblob[col_name] = df_textblob[col_name].astype('str')
                            def get_polarity(text):
                                return TextBlob(text).sentiment.polarity
                            df_textblob['Polarity'] = df_textblob[col_name].apply(get_polarity)

                            df_textblob.loc[df_textblob.Polarity>0,'Sentiment']='Positive'
                            df_textblob.loc[df_textblob.Polarity==0,'Sentiment']='Neutral'
                            df_textblob.loc[df_textblob.Polarity<0,'Sentiment']='Negative'
                            
                            df_textblob.columns = [col_name, 'Polarity', 'Sentiment']

                            tb_counts = df_textblob.Sentiment.value_counts().to_frame().reset_index()
                            tb_counts.columns = ["Sentiment", "Count"]

                            import plotly.express as px
                            fig = px.pie(tb_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: TextBlob Results')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.subheader("Resulting DataFrame")
                            st.write(df_textblob)
                            tmp_download_link = download_link(df_textblob, 'textblob_sentiment.csv', 'Download as CSV')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                            
                if type_task ==  "Vader": 

                            st.subheader("Vader")

                            sid = SentimentIntensityAnalyzer()

                            df_vader.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
                            df_vader[col_name] = df_vader[col_name].astype('str')
                            df_vader['scores'] = df_vader[col_name].apply(lambda x: sid.polarity_scores(x))
                            df_vader['compound'] = df_vader['scores'].apply(lambda score_dict: score_dict['compound'])

                            df_vader.loc[df_vader.compound>0,'Sentiment']='Positive'
                            df_vader.loc[df_vader.compound==0,'Sentiment']='Neutral'
                            df_vader.loc[df_vader.compound<0,'Sentiment']='Negative'
                            #df_vader['Sentiment'] = df_vader['Sentiment'].apply(str)
                            df_vader = df_vader[[col_name, 'scores', 'compound', "Sentiment"]]
                            df_vader.columns = [col_name, 'Scores', 'Compound', "Sentiment"]

                            vd_counts = df_vader.Sentiment.value_counts().to_frame().reset_index()
                            vd_counts.columns = ["Sentiment", "Count"]

                            import plotly.express as px
                            fig = px.pie(vd_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: Vader Results')
                            st.plotly_chart(fig, use_container_width=True)

                            st.subheader("Resulting DataFrame")
                            st.write(df_vader)
                            tmp_download_link = download_link(df_vader, 'vader_sentiment.csv', 'Download as CSV')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

            except KeyError:        
                st.warning('Please input the column name (case-sensitive).')        
                break
                    
            break
        except ValueError:
            st.warning('Please upload the csv file to proceed.')
            break
