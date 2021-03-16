from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
z1=np.random.randint(low=1,high=7, size=1000)
z2=np.random.randint(low=1,high=7, size=1000)
print(z1)
print(z2)
#Code for 2d histogram found online at matplotlib page just changed values for the dices
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
hist, xedges, yedges = np.histogram2d(z1, z2, bins=6, range=[[0.5, 6.5], [0.5, 6.5]])

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the 16 bars.
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

plt.show()
y=z1+z2
plt.hist(y, bins=np.arange(2, 14), align="left", rwidth=0.9)
plt.title("Y=z1+z2")
plt.show()
