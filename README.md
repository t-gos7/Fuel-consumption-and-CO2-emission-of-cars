## Predicting fuel consumption and CO2 emission of cars using Multiple Linear Regression

In this project, I have created a regression model which predicts CO2 emission and fuel consumption depending on
some other attributes. 

> ### Introduction to the dataset : 
The dataset `FuelConsumptionCo2.csv` has these following columns:
- **MODELYEAR** e.g. 2014
- **MAKE** e.g. Acura
- **MODEL** e.g. ILX
- **VEHICLE CLASS** e.g. SUV
- **ENGINE SIZE** e.g. 4.7
- **CYLINDERS** e.g 6
- **TRANSMISSION** e.g. A6
- **FUELTYPE** e.g. z
- **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
- **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
- **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
- **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0

> ### Loading data : 

```
''' Importing libraries '''
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
%matplotlib inline

# loading the dataset into an pandas.DataFrame object using pandas.read_csv() method
df = pd.read_csv("FuelConsumption.csv")
df.head() # first few rows of the dataset
```
Here I have taken only `ENGINESIZE, CYLINDERS, FUELCONSUMPTION_CITY, FUELCONSUMPTION_HWY` and `FUELCONSUMPTION_COMB` as independent variables and `CO2EMISSIONS` is dependent. Goal of this model is to predict `CO2EMISSIONS` for given values of other attrs.

```
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(5)
```

Now, lets check dependency of `CO2EMISSIONS` on `ENGINESIZE`:
```
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
```
![](https://github.com/t-gos7/Fuel-consumption-and-CO2-emission-of-cars/blob/master/EngineSize-vs-Emission.png)

Similarly, other variables choosed here will show linear dependency with the `CO2EMISSIONS` variable.

> ### Splitting training and testing data : 
There need a way to evaluate how good my model is, and for that we need data which was not used in training the model. That's why
splitting the dataset into two parts: one part will be used for training the model and other part for testing it. For splitting, I have
used boolean mask indexing on the dataframe to get `train`. The rest of it will be testing data `test`.

```
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
```
We need to check if the data is taken randomly and not taking a part of dataset in `train` data. To check that, plotting `CO2EMISSIONS` of train data with `ENGINESIZE`. If the output is not scattered over the dataset, then we need to run again. When I ran it, it produced 
the output as shown below:

```
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
```
![](https://github.com/t-gos7/Fuel-consumption-and-CO2-emission-of-cars/blob/master/TrainData-EngineSize-vs-Emission.png)

> ### Building the regression model
Here we have multiple independent variables which predicts the `CO2EMISSION`. We will use `linear_model.LinearRegression()` from
`sklearn` package to built the model.

```
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)
```
The *coefficient* and *intercept* are the parameters of the fit line. 
