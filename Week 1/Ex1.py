from scipy.stats import kurtosis
from scipy.stats import skew
import numpy as np
import matplotlib.pyplot as plt

#Για να έχουμε τα ίδια αποτελέσματα με την απεικόνιση στο report διαλέγω ένα σταθερό seed
np.random.seed(1)

def oneDieRolls(numOfThrows):
    sumOfThrows= np.random.randint(low=1,high=7, size=numOfThrows)
    return sumOfThrows


rolls=[20,100,1000]
listOfThrows=[]
for n in rolls:
    listOfThrows.append(oneDieRolls(n))

d = np.diff(np.unique(listOfThrows[0])).min()
left_of_first_bin =listOfThrows[0].min() - float(d)/2
right_of_last_bin = listOfThrows[0].max() + float(d)/2
fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()
ax0.hist(listOfThrows[0], np.arange(left_of_first_bin, right_of_last_bin + d, d),rwidth=0.5)
ax0.set_title('N=20')
ax1.hist(listOfThrows[1], np.arange(left_of_first_bin, right_of_last_bin + d, d),rwidth=0.5)
ax1.set_title('N=100')
ax2.hist(listOfThrows[2], np.arange(left_of_first_bin, right_of_last_bin + d, d),rwidth=0.5)
ax2.set_title('N=1000')

fig.tight_layout()
plt.show()


# EΡΩΤΗΜΑ Β

#calculate the N die rolls
nValues=[10,20,50,100,500,1000]
newListOfThrows=[]
np.random.seed(9990)
for n in nValues:
    newListOfThrows.append(oneDieRolls(n))

meanValue=[]
varianceValue=[]
skewnessValue=[]
kurtosisValue=[]
#Mean value calculation for diff N values (τα εχω σε μορφή πίνακα μήπως χρειαστεί κάποιο plot)
for i in range(len(newListOfThrows)):
    meanValue.append(np.mean(newListOfThrows[i]))
    varianceValue.append(np.var(newListOfThrows[i]))
    skewnessValue.append(skew(newListOfThrows[i]))
    kurtosisValue.append(kurtosis(newListOfThrows[i],fisher=False))
    print(20 * '==')
    print("FOR",nValues[i],"DIE ROLLS")
    print("Mean value: ",format(meanValue[i],'.4f')," Difference: ",format(abs(3.5-meanValue[i]),'.4f'))
    print("Variance: ",format(varianceValue[i],'.4f')," Difference: ",format(abs(2.9166-varianceValue[i]),'.4f'))
    print("Skewness is",format(skewnessValue[i],'.4f'),"Difference: ",format(abs(skewnessValue[i]),'.4f'))
    print("Kurtosis is",format(kurtosisValue[i],'.4f'),"Difference: ",format(abs(1.731428-kurtosisValue[i]),'.4f'))

plt.title("Mean Values for increasing N")
plt.plot(meanValue,'ro-', linewidth=2, markersize=12)
plt.show()