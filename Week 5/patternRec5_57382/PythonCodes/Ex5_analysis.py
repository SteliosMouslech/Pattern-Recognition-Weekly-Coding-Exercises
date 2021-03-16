import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
sns.set(color_codes=True)

iris = load_iris()

data = pd.DataFrame(data= np.c_[iris['data']],
                     columns= iris['feature_names'])
data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

a=sns.pairplot(data,hue='species')
plt.show()
