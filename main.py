import streamlit as st
import pandas as pd
import numpy as np
from os import path

import pickle
#st.title(" Hello world")
#st.write("This is my first PyCharm app")

#creating a dataframe
# df_Data = pd.DataFrame({'Column1': [1, 2, 3, 4],
#                         'Column2': ['a', 'b', 'c', 'd']})
#
# st.write(df_Data) #displaying the dataframe we created.


# st.title("iris dataset")
# df_iris = pd.read_csv(path.join("Data", "iris.csv"))
# st.write(df_iris)
# #filepath = Root/Dta/iris.csv
# #plotting a scatter plot using the data
# st.scatter_chart(df_iris[["sepal_length","sepal_width"]])

# st.title("My Favourite Place")
# df_map = pd.DataFrame(np.array([[25.342751359185144, 55.48976339462592]]),
#                      columns= ["lat","lon"])
# st.map(df_map)

# petal_length = st.slider("Please choose a petal length", min_value=1, max_value=6)
# petal_width = st.slider("Please choose a petal width")
# sepal_length = st.slider("Please choose a sepal length")
# sepal_width = st.slider("Please choose a sepal width")
#
st.title("Flower species predictor")
petal_length = st.number_input("Please choose a Petal Length", placeholder="please enter a valid number b/w 1.0 and 6.9", min_value =1.0, max_value=6.9, value=None)
petal_width = st.number_input("Please choose a petal width" , placeholder="please enter a valid number b/w 0.1 and 2.5", min_value =0.1, max_value=2.5, value=None)
sepal_length = st.number_input("Please choose a sepal length", placeholder="please enter a valid number b/w 4.3 and 7.9", min_value =4.3, max_value=7.9, value=None)
sepal_width = st.number_input("Please choose a sepal width", placeholder="please enter a valid number b/w 2.0 and 4.4", min_value =2.0, max_value=4.4, value=None)

#prepare dataframe for prediction
df_user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
      #using the pkl file, creating an ML model named 'iris predictor'
model_path = path.join("model","iris_classifier.pkl")
with open(model_path, 'rb') as file:
    iris_predictor = pickle.load(file)


    dict_species={0:'setosa',1: 'versicolor',2: 'virginica'}
# iris_predictor = pickle.load(path.join("model","iris_classifier.pkl"))
# st.write(df_user_input)
if st.button("Predict Species"):
   if((petal_length==None) or (petal_width==None)
           or (sepal_length==None) or (sepal_width==None)):
       #will be executed when any of the value is not entered properly
       st.write("Please fill all values")
   else:
       #prediction can be done here
       predicted_species = iris_predictor.predict(df_user_input)
       #predicted_species[0] will give us the value in the dataframe
       #we use that value to find the corresponding species from the dictionary 'dict_species'
       st.write("the Species is:",dict_species[predicted_species[0]])
#
