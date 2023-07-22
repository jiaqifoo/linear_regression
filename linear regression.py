import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
from matplotlib import pyplot
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
original = pd.read_csv('student-mat.csv',sep=";")

print(original.info())
#No missing values found

#picking attributes which i assume are highly related with G3
data = original[['school', 'sex','reason','studytime','internet', 'romantic','famrel','health','absences','failures','paid','higher','G1','G2','G3']]
data = pd.get_dummies(data, columns=['school', 'sex','paid', 'higher', 'internet', 'romantic', 'reason'], drop_first=True, dtype=int)

plt.figure(figsize=(10,5))
sns.heatmap(data.corr(), annot=True)
plt.show()

#picking out attributes that are correlated to G3 by at least 0.098
new = original[['sex','paid','higher','studytime','internet','G1','G2','G3']]
new = pd.get_dummies(new, columns=['sex','paid','higher','internet'])

x=new.drop(['G3'],axis=1)
y=new['G3']

print("Shape of X is: ", x.shape)
print("Shape of y is: ", y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=423)

linearmodel = linear_model.LinearRegression()
linearmodel.fit(x_train,y_train)
print(linearmodel.score(x_test,y_test))

predictions=linearmodel.predict(x_test)

print(predictions)