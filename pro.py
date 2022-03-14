import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#matplotlib inline
HouseDF = pd.read_csv('USA_Housing.csv')
sns.pairplot(HouseDF)
sns.heatmap(HouseDF.corr(), annot=True)
X = HouseDF.iloc[: , :-2]
y = HouseDF.iloc[:, -2]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

sns.displot((y_test-predictions),bins=50)
