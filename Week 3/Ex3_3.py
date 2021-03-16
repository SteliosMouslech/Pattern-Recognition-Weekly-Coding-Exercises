import scipy.integrate as integrate
import numpy as np
import math
import matplotlib.pyplot as plt

ptheta0= lambda theta: np.sin(math.pi*theta)
A=1/integrate.quad(ptheta0,0,1)[0]
print("A is: ",A)

#kefali 1 grammata 0
d=np.array([1,1,0,1,0,1,1,1,0,1])


y=[]
for i in[1,5,10]:
    k=np.count_nonzero(d[0:i])
    numerator= lambda theta: (theta**k)*(1-theta)**(i-k)*A*ptheta0(theta)
    denominator=integrate.quad(numerator,0,1)[0]
    theta=np.linspace(0,1,1000)
    y.append(numerator(theta)/denominator)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['bottom'].set_position('zero')
plt.plot(theta,y[0], 'r',label="P(theta|D1)")
plt.plot(theta,y[1], 'g',label="P(theta|D5)")
plt.plot(theta,y[2], 'b',label="P(theta|D10)")
plt.legend()
plt.show()


print("the max value of P(θ|D10) is:",max(y[2]))
print("for x value",(np.argmax(y[2]))/1000)



