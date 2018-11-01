import matplotlib.pyplot as plt
import numpy as np

a = np.loadtxt("Trainsteps-NoiseCurve.txt", comments='#')

plt.plot(a[0:15,1], a[0:15,2], label="Noise 1.5")
plt.plot(a[16:31,1], a[16:31,2], label="Noise 3.0")
plt.plot(a[32:47,1], a[32:47,2], label="Noise 5.0")

plt.xlabel("Training steps")
plt.ylabel("Accuracy")

plt.legend(loc='upper left')

plt.show()