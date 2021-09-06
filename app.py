#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import streamlit as st
from sentiment_analyzer import sentiment_analysis, get_noun, top_neg_word, frequency_counter
from text_preprocessing import clean_review
import io
import base64
import os
import json
import pickle
import uuid
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit.components.v1 as components
import time
#os.environ['DISPLAY']= ':0'
import pyautogui
import base64
LOGO_IMAGE = "./Logo.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
    }
    .logo-img {
        
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}" width="160" height="100">
       
    </div>
    """,
    unsafe_allow_html=True
)



st.title("Patient Review Sentiment Analyzer")
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            #object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link



    
uploaded_file = st.file_uploader("Upload Input Data File")
if uploaded_file is None:
    st.error("Please upload file")
if uploaded_file is not None:
    st.success("File Uploaded Successfully")
    df = pd.read_csv(uploaded_file, encoding ='latin1')
    df.columns =['Reviews']
    list_text = df.Reviews.tolist()
    
    left, centre, right = st.columns(3)
    with left:
        st.write('Step 1')
    with centre:
        st.write('Text Processing')
    with right:
        process = st.button('Process input reviews')
        
    
    if process:
        st.write(f"Sample Processed Text : {clean_review(list_text[0])}")
        
    
    analysis = sentiment_analysis(list_text)
    

    left, centre, right = st.columns(3)
    left.write('Step 2')
    centre.write('Sentimental Analysis')
    with right:
        sentiments = st.button('Generate Sentiments')
    if sentiments:
        sent_dict = dict(zip(analysis.Reviews, analysis.Sentiment))
        vAR_firstSent= next(iter((sent_dict.items())) )
        st.write(f" {vAR_firstSent[0]} : {vAR_firstSent[1]}")
        #st.write(first_pair[1])
        
        
 
    left, centre, right = st.columns(3)
    left.write('Step 3')
    centre.write('Rank Sentiments')
    with right:
        rank = st.button('Generate Ranking')
     
    if rank:
        rank_dict = dict(zip(analysis.Reviews, analysis.Ranking))
        vAR_firstRank= next(iter((rank_dict.items())) )
        #st.write(f" {vAR_firstRank[0]} : {vAR_firstRank[1]}")
        #st.write(rank_dict)
    
    left, centre, right = st.columns(3)
    left.write('Step 4')
    centre.write('Visualization')
    with right:
        wordcloud = st.button("Generate Wordcloud")
        
    if wordcloud:
        clean_text = []
        for text in list_text:
            text = clean_review(text)
            clean_text.append(text)
            textss = ' '.join(map(str, clean_text))
        nouns = get_noun(clean_text)
        #only_neg = top_neg_word(clean_text)
        #print(f'Negative word: {top_neg_word(clean_text)}')
        #print(f'Cause of negativity: {nouns}')
        neg_words = ' '.join(map(str, nouns ))
        #only_neg_words = ' '.join(map(str, only_neg))
        fig, ax = plt.subplots()
        wordcloud =  WordCloud().generate(neg_words)
        plt.imshow(wordcloud,interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
        
    left, centre, right = st.columns(3)
    left.write('Step 5')
    centre.write('Visualization')
    with right:
        freq_chart = st.button("Generate Word Frequency Chart")
    if freq_chart:
        freq = frequency_counter(list_text)
        fig = px.bar(freq, x="count", y="word", height=1000)
        #fig, axes = plt.subplots(3,1,figsize=(8,20))
        st.plotly_chart(fig)        
        
        
    
    left, centre, right = st.columns(3)
    left.write('Step 6')
    centre.write('Model Outcome')
    with right:
        outcome_df = st.button("View Model Outcome")
        
    if outcome_df:
        st.write(analysis)
        
       
    filename = 'model_outcome.xlsx'
    download_button_str = download_button(analysis, filename, f' Download Model Outcome', pickle_it=False)
    outcome = st.markdown(download_button_str, unsafe_allow_html=True) 
   
    
    st.markdown('<h2> <br><font style="color: #5500FF;">Powered by Google Cloud & Colab</font></h2>',unsafe_allow_html=True)
    st.markdown('<hr style="border-top: 6px solid #8c8b8b; width: 150%;margin-left:-180px">',unsafe_allow_html=True) 

    st.sidebar.selectbox("",
       ('Libraries Used','pandas','numpy', 'sklearn','nltk', 'streamlit','wordcloud', 'matplotlib'))
    
    st.sidebar.selectbox("",('GCP Services Used','VM Instance','Compute Engine'))
    
    st.sidebar.button("Refresh"):
        
    
    
    
    
      
        






