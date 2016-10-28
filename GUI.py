import matplotlib.pyplot as plt
import matplotlib.path as mpath
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

N = 100
data = np.random.random((N,7))
x = data[:,0]
y = data[:,1]
point = data[:,2:4]
rgb = plt.get_cmap('jet')(100)

ax.scatter(x,y, color = rgb)

plt.show()