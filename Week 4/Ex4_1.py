import numpy as np
import Ex4_myFunctions as ex4
import matplotlib.pyplot as plt



np.random.seed(1334)


h = [0.05, 0.2]
sampleSize = [32, 256, 5000]
listofPlots=[]
for i in range(len(h)):
    for j in range(len(sampleSize)):
        samples = np.random.uniform(0, 2, size=sampleSize[j])
        px=ex4.parzen_gauss_kernel(samples,h[i],-1,3)
        x = np.arange(-1, 3, h[i])
        listofPlots.append((x,px))


fig, axs = plt.subplots(3,2,sharey=True)
axs[0, 0].plot(listofPlots[0][0], listofPlots[0][1])
axs[0, 0].set_title('h=0.05 N=32')
axs[1, 0].plot(listofPlots[1][0], listofPlots[1][1], 'tab:orange')
axs[1, 0].set_title('h=0.05 N=256')
axs[2, 0].plot(listofPlots[2][0], listofPlots[2][1], 'tab:green')
axs[2, 0].set_title('h=0.05 N=5000')
axs[0, 1].plot(listofPlots[3][0], listofPlots[3][1], 'tab:red')
axs[0, 1].set_title('h=0.2 N=32')
axs[1, 1].plot(listofPlots[4][0], listofPlots[4][1], 'tab:red')
axs[1, 1].set_title('h=0.2 N=256')
axs[2, 1].plot(listofPlots[5][0], listofPlots[5][1], 'tab:red')
axs[2, 1].set_title('h=0.2 N=5000')





k=[32,64,256]

listofPlots2=[]
for i in range(len(k)):
    px=ex4.knn_density_estimate(samples,k[i],-1,3,0.1)
    listofPlots2.append(px)

fig2, axs2 = plt.subplots(3,1,sharey=True)

x = np.arange(-1, 3, 0.1)
axs2[0].plot(x, listofPlots2[0])
axs2[0].set_title('k=32 N=5000')
axs2[1].plot(x, listofPlots2[1], 'tab:orange')
axs2[1].set_title('k=64 N=5000')
axs2[2].plot(x, listofPlots2[2], 'tab:green')
axs2[2].set_title('k=256 N=5000')

plt.show()


