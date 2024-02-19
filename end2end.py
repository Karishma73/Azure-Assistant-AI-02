#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
import os
import env
#nltk.download()
#import tkinter as tk
#from tkinter.ttk import *
from pandasai import SmartDatalake
from pandasai.llm import AzureOpenAI
import streamlit as st
import matplotlib
from textblob import TextBlob
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
st.title("Project Nairobi powered by GenAl")
os.environ["api_token"] = env.api_token
os.environ["api_base"] = env.api_base
os.environ["api_version"] = env.api_version
os.environ["deployment_name"] = env.deployment_name
llm = AzureOpenAI(
    api_token=st.secrets["api_token"],
    api_base=st.secrets["api_base"],
    api_version=st.secrets["api_version"],
    deployment_name=st.secrets["deployment_name"]
)

#Data = pd.read_csv('v1_final_data.csv')
#st.title("Start your work....")
user_input = st.file_uploader("Please upload your csv file here ", type=['csv'])
if user_input is not None:
    Data = pd.read_csv(user_input)
    data = Data.copy(deep=True)
    data = data.dropna()
    #data[['this_month','last_month']] = data[['this_month','last_month']].apply(pd.to_datetime)
    st.write(data.head())
    st.write(data.info())

    user_question = st.text_area("Enter your question here :")
    if st.button("Answer"):
        if user_question:
            with st.spinner('LLM is working on your question'):
            #st.write("LLM model is predicting your answer.......")
            #dl = SmartDatalake([df], config={"llm": llm})
              df = SmartDatalake([data], config={"llm": llm})
            #ans = df.chat(user_question)
            #print(ans)
              st.write(df.chat(user_question))
            #st.image(img)

        else:
            st.warning("Please enter your question.")


st.title("We are still working on it ...")



