import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df=pd.read_csv('iris.csv')

headers=["sepal-length","sepal-width","petal-length","petal-width","class"]

df.columns=headers

print(df.head(5))
print(df.describe())

print(df.tail(5))
print(df.shape)
print(df.dtypes)

# Histogram
df.hist()
plt.show()

# Boxplot
df.boxplot()
plt.show()

# Histograms of Attributes
df.plot(kind='hist', subplots=True, layout=(2,2), sharex=False, sharey=False,title="iris dataset histogram")
plt.show()

# Scatter Matrix
scatter_matrix(df)
plt.show()
