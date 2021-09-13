import streamlit as st
from src.keyword_ex import *
from src.text_summarizer import *
from src.pos_ner import *
import en_core_web_sm

if __name__ == '__main__':


    st.markdown("<h1 style='text-align: center;'>Natural Language Processing</h1>", unsafe_allow_html=True)
    st.markdown("<span style=“background-color:#121922”>",unsafe_allow_html=True)

    menu = ["Keyword Extraction", "Information Extraction", "Text Summarization"]
    choice = st.sidebar.selectbox("Natural Language Processing", menu)

    if choice == "Keyword Extraction":

        st.markdown("<h3 style='text-align: center;'>Unsupervised Keyword Extraction: spaCy-PyTextRank</h3>", unsafe_allow_html=True)
        spaCy()

    if choice == "Information Extraction":
        posner()

    if choice == "Text Summarization":

        st.markdown("<h3 style='text-align: center;'>Extractive Text Summarization: spaCy</h3>", unsafe_allow_html=True)

        no_sent = st.number_input("How many sentences would you like in the summarized text?")
        raw_text = st.text_area("Your Text")
        if st.button("Show Summary", key = "321"):
            summary = spacy_sum(raw_text, no_sent)
            st.info(summary)
                                
            tmp_download_link = download_link(summary, 'summary_spacy.txt', 'Download Text')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
