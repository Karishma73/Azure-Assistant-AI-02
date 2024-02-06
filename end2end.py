#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
#nltk.download()
import Tkinter as tk
from pandasai import SmartDatalake
from pandasai.llm import AzureOpenAI
import streamlit as st
import matplotlib
from textblob import TextBlob
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
st.title("Project Nairobi powered by GenAl")
# In[2]:
llm = AzureOpenAI(
    api_token="63efcdcc12ea499e85cc622b4c35a921",
    api_base="https://datascience-dall-e-test.openai.azure.com/",
    #api_version="2023-05-15",
    api_version="2023-07-01-preview",
    deployment_name="Test-for-Karishma"
)

#Data = pd.read_csv('v1_final_data.csv')
#st.title("Start your work....")
user_input = st.file_uploader("Please upload your csv file here ", type=['csv'])
if user_input is not None:
    #Data = pd.read_csv(r"C:\Users\Karishma.Nanda\Documents\ofc_project\nairobi\v1_final_data.csv")
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


# In[3]:




# In[4]:





# In[5]:


#data.sort_values(by=['click_through_rate'], ascending=False)


# In[6]:


#df.chat("calculate the sum of open_rate and click_rate for subject fastfix flex documentatie ") #very good

#df.chat('what is open_rate in month November')# very good
# In[7]:


#response = df.chat(
   # "Plot the histogram of first five subject showing for each em_score_r, using different colors for each bar")
#print(response)


# In[8]:


#data.info()


# In[9]:


#data["date"] = data["date"].apply(pd.to_datetime)


# In[10]:


st.title("Karishma is still working on it ...")
# In[ ]:





# In[ ]:





# In[ ]:




