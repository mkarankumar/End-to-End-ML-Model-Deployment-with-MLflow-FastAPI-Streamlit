from sklearn.datasets import load_iris
import pandas as pd

def data_base():
    data=load_iris()
    x=data.data 
    y=data.target
    return x,y 


user_db=pd.DataFrame(columns=["Sepal length","sepal widtgh","petal width","patal length"])

def user_data(input):
    global user_db 
    user_db.iloc[len(user_db)]=input 
    return "The value has been upated" 
