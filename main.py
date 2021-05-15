import pandas  as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import numpy as np

Physics_Data = {
  "Lepton Universality": [1.0,2.6,3.1],#lepton universality statiscal deviation
  "Magnetic Moment": [3.3,3.7,4.2], #muon magnetic moment
  "Time":[0,1,2],

}
#Time is an arbitrary legnth representing order in which the discoveries were made
physics_df = pd.DataFrame(data=Physics_Data)

print(physics_df.head())

x = physics_df[["Time","Time"]]
y = physics_df[["Lepton Universality","Magnetic Moment"]]

multiple = linear_model.LinearRegression(fit_intercept = True, normalize = True)
multiple.fit(x,y)
prediction = multiple.predict(x)
plt.plot(x,prediction,color="red")
plt.scatter(x,y)
plt.xlabel('Time')
plt.ylabel('Statistical Deviation')
plt.show()

print('Our multiple linear model had an R^2 of: %0.3f'%multiple.score(x, y))

physics_df['Prediction'] = prediction

