#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("Dataset1.csv")
df


# In[2]:


with open("dashboard.py","r")as f:
    print(f.read())


# In[3]:


import os
print("Jupyter folder:")
print(os.getcwd())

print("\nFiles here:")
print(os.listdir())


# In[4]:


print(os.path.abspath("dashboard.py"))


# In[11]:


with open(r"C:\Users\New\Dashboard.py", "w") as f:
    f.write("""import streamlit as st
import pandas as pd

st.title("Train Journey Prediction Dashboard")

df = pd.read_csv("Dataset1.csv")

st.success("Dataset loaded successfully!")

st.subheader("Dataset Preview")
st.dataframe(df.head(10))

st.subheader("Dataset Summary")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(df))
col2.metric("Total Stations", df["Station_Name"].nunique())
col3.metric("Total Trains", df["Train_No"].nunique())

st.subheader("Distance Analysis")
st.bar_chart(df["Distance"].head(20))

st.subheader("Journey Duration Prediction")

distance = st.number_input("Enter Distance (km)", min_value=0.0, value=100.0)

if st.button("Predict Journey Duration"):
    duration = distance / 60
    st.success("Predicted Journey Duration: {:.2f} hours".format(duration))
""")

print("Dashboard code saved!")


# In[8]:


with open(r"C:\Users\new\dashboard.py","r")as f:
    print(f.read())


# In[12]:


import os
print(os.path.exists(r"C:\Users\new\Dataset1.csv"))


# In[ ]:


get_ipython().system('streamlit run dashboard.py')


# In[ ]:




