import streamlit as st
import pandas as pd
import base64
import spacy
import pytextrank

def input():
    raw_text = st.text_area("Your Text")
    return raw_text

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'  


def spacy_pytextrank(text):
    
    # load a spaCy model, depending on language, scale, etc.
    nlp = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    doc = nlp(text)

    text = []
    rank = []
    count = []
    chunks = []

    # examine the top-ranked phrases in the document
    for phrase in doc._.phrases:
        text.append(phrase.text)
        rank.append(phrase.rank)
        count.append(phrase.count)
        chunks.append(phrase.chunks)
        
    df = pd.DataFrame()
    df['Keyword'] = text
    df['Rank'] = rank
    df['Count'] = count
    df['Chunks'] = chunks
    df['Chunks'] = df['Chunks'].apply(str)

    df = df.sort_values('Count', ascending = False)

    return df


def spaCy():

    #st.subheader("spaCy-PyTextRank")
    raw_text = input()
    if st.button("Show Keywords"):
        df = spacy_pytextrank(raw_text)
        st.dataframe(df)
        tmp_download_link = download_link(df, 'spaCy_PyTextRank.csv', 'Download as CSV')
        st.markdown(tmp_download_link, unsafe_allow_html=True)