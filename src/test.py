import numpy as np
a=[1,2,3,4]
mean = np.mean(a)
print ("this is the mean", mean)

msg = "hello world"
print (msg)

import matplotlib.pyplot as plt
import matplotlib as mlp

x = np.linspace(0,20,100)
plt.plot(x, np.sin(x))
plt.show()