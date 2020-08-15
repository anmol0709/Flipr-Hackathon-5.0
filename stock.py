import time
import os
import pandas as pd
import csv
import numpy as np
import glob
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

start=time.time()
path=os.path.abspath(os.curdir)

#Time Series Model LSTM
def lstm(df):
	scaler = MinMaxScaler()
	train_open_scaled= scaler.fit_transform(df)
	xtrain=[]
	ytrain=[]
	for i in range(5,len(train_open_scaled)):
	    xtrain.append(train_open_scaled[i-5:i,0])
	    ytrain.append(train_open_scaled[i,0])

	xtrain, ytrain = np.array(xtrain), np.array(ytrain)
	xtrain= np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))
	model=Sequential()
	model.add(LSTM(50,activation='relu',input_shape=(5,1)))
	model.add(Dense(1))
	model.compile(optimizer='adam',loss='mse')
	model.fit(xtrain,ytrain,epochs=100,verbose=0)
	# print(df.iloc[0])
	return df

#Reading xlsx files
#Converting xlsx to csv
df=pd.read_excel("Train_dataset.xlsx")
df.to_csv("Train_dataset.csv",sep=",")

Train=pd.read_csv("Train_dataset.csv",low_memory=False).drop(['Unnamed: 0'],axis=1)
#Turning off warnings
half_count=len(Train)/2
Train= Train.dropna(thresh=half_count,axis=1) 

warnings.filterwarnings("ignore")
#Storing stock index
ind=Train['Stock Index']
Train=Train.drop('Stock Index',axis=1)
#Splitting dataset into target and result
Y=Train['Stock Price']
X=Train.drop('Stock Price',axis=1)
#We need to convert certain columns to float for prediction 
convt_columns=['Index','Industry']
#Converting string to float
le=LabelEncoder()
for col in convt_columns:
	le.fit(X[col].astype(str))
	X[col]=le.transform(X[col].astype(str))

# Filling of empty values
X=X.fillna(round(X.mean(),2))
Y=Y.fillna(round(Y.mean(),2))

#####################################################################################################
#Splitting data into train and test 
Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,test_size=0.20,random_state=30)
#Transorming data for SVM model
scaler=StandardScaler()
scaler.fit(Train_X)
Train_X=scaler.fit_transform(Train_X)
Test_X=scaler.fit_transform(Test_X)

#Training the model 
from sklearn.svm import SVR
model=SVR(kernel='rbf')
model.fit(Train_X,Train_Y)
pred=model.predict(Test_X)
print("Model: SVM \n")
print("Score :",round(model.score(Test_X,Test_Y),4))
print("Mean absolute error :",round(metrics.mean_absolute_error(Test_Y,pred),2))
print("_____________________________\n")
#####################################################################################################
# Uploading test dataset
df=pd.read_excel('Test_dataset.xlsx')
df.to_csv("Test_dataset.csv",sep=",")
Test=pd.read_csv("Test_dataset.csv",low_memory=False).drop(['Unnamed: 0'],axis=1)
id_Test=Test['Stock Index']
Test=Test.drop('Stock Index',axis=1)
#We need to convert certain columns to float for prediction 
convt_columns=['Index','Industry']
#Converting string to float
le=LabelEncoder()
for col in convt_columns:
	le.fit(Test[col].astype(str))
	Test[col]=le.transform(Test[col].astype(str))
#Filling of empty values
Test=Test.fillna(round(Test.mean(),4))
Test.to_csv("Test_dataset.csv")

scaler=StandardScaler()
scaler.fit(Test)
Test=scaler.fit_transform(Test)
#####################################################################################################
#Part-01
prediction=model.predict(Test)
Pred_stock_price_10th=pd.DataFrame(prediction)
Pred_stock_price_10th.index=id_Test
Pred_stock_price_10th.columns=['Stock Price']
print("Stock Price on 10th August")
print(Pred_stock_price_10th)
Pred_stock_price_10th.to_csv("outputfile_01.csv")
print("_____________________________\n")
#####################################################################################################
#Part-02
#First predict p/c ratio on 16th August and then replace it with the value in test dataset and make prediction
#Predicting value on 16th August
Data=glob.glob('Test_dataset.xlsx')
for excel in Data:
	sheet=excel.split('.')[0]+'.csv'
	pc=pd.read_excel('Test_dataset.xlsx',sheet_name='Put-Call_TS',index=False)
	pc.to_csv('Time_series.csv')

Train_02=pd.read_csv('Time_series.csv',low_memory=False).drop(['Unnamed: 0'],axis=1)
Train_02=Train_02.drop('Stock Index',axis=1)
Train_02=Train_02.fillna(method='bfill')

Train_02=pd.DataFrame(Train_02)
Train_02.to_csv('Time_series.csv',header=False)
#Removing unnecessary header
Train_02=pd.read_csv('Time_series.csv',low_memory=False)
#####################################################################################################

dates=["2020-08-10 00:00:00","2020-08-11 00:00:00","2020-08-12 00:00:00","2020-08-13 00:00:00","2020-08-14 00:00:00","2020-08-15 00:00:00"]
i=0;
#Train LSTM model and predict put-call ratio on 16th August 
val=[]
for index,row in Train_02.iterrows():
	if i==1:
		break
	df=[]
	df.append(row["2020-08-10 00:00:00"])
	df.append(row["2020-08-11 00:00:00"])
	df.append(row["2020-08-12 00:00:00"])
	df.append(row["2020-08-13 00:00:00"])
	df.append(row["2020-08-14 00:00:00"])
	df.append(row["2020-08-15 00:00:00"])
	df=pd.DataFrame(df)
	df.index=dates
	a=lstm(df)
	i+=1;

df=[]
df.append(Train_02["2020-08-10 00:00:00"])
df.append(Train_02["2020-08-11 00:00:00"])
df.append(Train_02["2020-08-12 00:00:00"])
df.append(Train_02["2020-08-13 00:00:00"])
df.append(Train_02["2020-08-14 00:00:00"])
df.append(Train_02["2020-08-15 00:00:00"])
df=pd.DataFrame(df)
df.index=dates
val=df.mean()
print("_____________________________\n")
# val.append(lstm(df))
Train_02.insert(7,'2020-08-16 00:00:00',val)
Train_02.to_csv('Time_series.csv')
#####################################################################################################
pc_16th=Train_02['2020-08-16 00:00:00']

# Replacing p/c ratio in test dataset with value on 16th August
Test_02=pd.read_csv('Test_dataset.csv',low_memory=False).drop(['Unnamed: 0'],axis=1)
Test_02=Test_02.drop('Put-Call Ratio',axis=1)
Test_02.insert(12,'Put-Call Ratio',pc_16th)

# Prediction
Test_02=scaler.fit_transform(Test_02)
predict_16th=model.predict(Test_02)

Pred_stock_price_16th=pd.DataFrame(predict_16th)
Pred_stock_price_16th.index=id_Test
Pred_stock_price_16th.columns=['Stock Price']
print("Stock Price on 16th August")
print(Pred_stock_price_16th)
Pred_stock_price_16th.to_csv("outputfile_02.csv")

end=time.time()
print("Execution Time",round(end-start,2))
print("_____________________________\n")