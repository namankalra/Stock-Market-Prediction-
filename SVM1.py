import numpy as np
import pandas as pd

df = pd.read_csv('STOCKS.csv')

df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

from datetime import date
f_date = date(2008, 8, 8)
l_date = date(2016, 7, 1)
delta = l_date - f_date
print(delta.days)

NoOfDays=[]

for i in range(0,1989):
        NoOfDays.append(i+1)
        
new_data['Days']=NoOfDays

new_data.drop('Date', axis=1, inplace=True)


"""scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(new_data)"""

#Convert the Date into no of days and then apply svm
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(new_data['Days'],new_data['Close'],random_state=42,
                                               test_size=0.2,shuffle=False)

x_train=X_train.values
y_train=Y_train.values
x_test=X_test.values
y_test=Y_test.values

x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)
y_test=y_test.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
Sc_X=StandardScaler()
Sc_Y=StandardScaler()
x_train=Sc_X.fit_transform(x_train)
y_train=Sc_Y.fit_transform(y_train)

from sklearn.svm import SVR

#Gaussian kernel
svr_rbf = SVR(kernel='rbf')

svr_rbf.fit(x_train,y_train)

y_rbf=Sc_Y.inverse_transform(svr_rbf.predict(Sc_X.transform(x_test)))

import matplotlib.pyplot as plt  
plt.scatter(x_test,y_test, label= "stars", color= "green",  
            marker= "*", s=30) 
plt.scatter(x_test,y_rbf, label= "stars", color= "red",  
            marker= "*", s=30) 

plt.xlabel('Days')
plt.ylabel('Value') 

plt.title('Stock Market Prediciton')

plt.legend() 

plt.show() 
