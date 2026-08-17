#!/usr/bin/env python
# coding: utf-8

# LEVEL:1 UNDERSTANDING THE DATA

# In[4]:


import pandas as pd
df=pd.read_csv("Dataset1.csv")
df


# In[6]:


#Total records and columns

print("Total records:",df.shape[0])
print("Total columns:",df.shape[1])

print("\ncolumn names:")
print(df.columns.tolist())


# In[7]:


# Train-wise starting and ending stations

df = df.sort_values(["Train_No", "Route_Number"])

train_table = df.groupby("Train_No").agg(
    Starting_Station=("Station_Name", "first"),
    Ending_Station=("Station_Name", "last")
).reset_index()

display(train_table)


# In[8]:


#Distance statistics

print("Distance statistics:")
display(df["Distance"].describe())


# In[9]:


#number of stops
stops=df.groupby("Train_No")["Station_Name"].count() - 1
print("Number of stops:")
print(stops.describe())


# In[10]:


# missing values
print("Missing values in each column:")
print(df.isnull().sum())


# In[11]:


# duplicate values
print("Number of duplicate rows:")
df.duplicated().sum()


# In[12]:


#checking incorrect distance values
print(df["Distance"].min())


# LEVEL:2 DATA CLEANING AND FEATURE CREATION

# In[13]:


#remove duplicate rows
print("Duplicates removed")
print("Total rows:",len(df))


# In[14]:


#checking missing values
print(df.isnull().sum())


# In[15]:


print(df.columns.tolist())
display(df[["Arrival_time","Departure_Time"]].head())


# In[16]:


# Convert time into minutes

df["Departure_Minutes"] = (
    pd.to_datetime(df["Departure_Time"], format="%H:%M:%S").dt.hour * 60
    + pd.to_datetime(df["Departure_Time"], format="%H:%M:%S").dt.minute
)

df["Arrival_Minutes"] = (
    pd.to_datetime(df["Arrival_time"], format="%H:%M:%S").dt.hour * 60
    + pd.to_datetime(df["Arrival_time"], format="%H:%M:%S").dt.minute
)

print(df.head())


# In[17]:


print(df[["Departure_Time","Departure_Minutes"]].head())
print(df[["Arrival_time","Arrival_Minutes"]].head())


# In[18]:


#total journey duration
df["Journey_Duration"]=df["Arrival_Minutes"]-df["Departure_Minutes"]

df.loc[df["Journey_Duration"]<0,"Journey_Duration"]+=1440
print(df["Journey_Duration"].head(10)) 


# In[19]:


#total nmbr of stops
df["Total_stops"]=df.groupby("Train_No")["Station_Name"].transform("count")-1
print(df[["Train_No","Station_Name", "Total_stops"]].head())


# LEVEL 3: FEATURE ANALYSIS WITH VISUALS

# In[20]:


#visualize distane vs journey duration
import matplotlib.pyplot as plt
plt.scatter(df["Distance"],df["Journey_Duration"])
plt.xlabel("Distance")
plt.ylabel("Journey Duration")
plt.title("Distance vs Journey Duration")
plt.show()


# In[21]:


#nmbr of stops vs journey duration
plt.scatter(df["Total_stops"],df["Journey_Duration"])
plt.xlabel("Total_stops")
plt.ylabel("Journey Duration")
plt.show()


# In[22]:


#correlation btwn features and journey duration
df[["Distance", "Total_stops","Journey_Duration"]].corr()


# In[23]:


#pivot table for nmbr of stops by train 
pd.pivot_table(df,values="Total_stops",index="Train_No",aggfunc="mean")


# LEVEL: 4 MODEL TRAINING AND EVALUATION

# In[24]:


#create journey duration

df["Journey_Duration"]=df["Arrival_Minutes"]-df["Departure_Minutes"]
print(df[["Departure_Time","Arrival_time","Journey_Duration"]].head())


# In[25]:


#split the dataset into training and testing sets 
from sklearn.model_selection import train_test_split
x=df[["Distance","Total_stops"]]
y=df["Journey_Duration"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Training data:",x_train.shape)
print("Testing data:",x_test.shape)


# In[27]:


#train a linear regression model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
print("Model trained successfully")


# In[30]:


#evaluate the model using MAE & RMSE
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
y_pred=model.predict(x_test)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

print("MAE:",mae)
print("RMSE:",rmse)


# In[31]:


#actual vs predicted journey duration
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Journey Duration")
plt.ylabel("Predicted Journey Duration")
plt.title("Actual vs Predicted Journey Duration")
plt.show()


# LEVEL: 5 MODEL COMPARISON AND USER TESTING

# In[32]:


#train a basic model using only one input feature
x_basic=df[["Distance"]]
y=df["Journey_Duration"]
x_train_basic,x_test_basic,y_train_basic,y_test_basic=train_test_split(x_basic,y,test_size=0.2,random_state=42)

basic_model=LinearRegression()
basic_model.fit(x_train_basic,y_train_basic)
print("Basic model trained successfully")


# In[34]:


#train an improved model using multiple input features
x_improved=df[["Distance","Total_stops"]]
y=df["Journey_Duration"]

x_train_improved,x_test_improved,y_train_improved,y_test_improved=train_test_split(x_improved,y,test_size=0.2,random_state=42)

improved_model=LinearRegression()
improved_model.fit(x_train_improved,y_train_improved)
print("Improved model trained successfully")


# In[36]:


#Compare both models using MAE and RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Predictions
basic_pred = basic_model.predict(x_test_basic)
improved_pred = improved_model.predict(x_test_improved)

# Basic model errors
basic_mae = mean_absolute_error(y_test_basic, basic_pred)
basic_rmse = np.sqrt(mean_squared_error(y_test_basic, basic_pred))

# Improved model errors
improved_mae = mean_absolute_error(y_test_improved, improved_pred)
improved_rmse = np.sqrt(mean_squared_error(y_test_improved, improved_pred))

print("Basic Model")
print("MAE:", basic_mae)
print("RMSE:", basic_rmse)

print("\nImproved Model")
print("MAE:", improved_mae)
print("RMSE:", improved_rmse)


# In[38]:


#Select the better model
if improved_mae < basic_mae and improved_rmse < basic_rmse:
    print("Improved Model performs better")
else:
    print("Basic Model performs better")


# LEVEL: 6 FINAL INTREACTIVE MACHINE LEARNING PROJECT

# In[40]:


# Level 6: Interactive Journey Duration Prediction

print("===== Journey Duration Prediction =====")

# Get input from user
distance = float(input("Enter Distance: "))
stops = int(input("Enter Total Stops: "))

# Create input data
new_data = [[distance, stops]]

# Predict journey duration
prediction = model.predict(new_data)

print("\nPredicted Journey Duration:", prediction[0], "minutes")

