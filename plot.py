import numpy as np
from matplotlib import pyplot as plt

name = 'Dim '
num = list(map(str, range(1,10)))
label = list(map(lambda y:name+y, num))
data = np.load('dataFile.npy')
print(data.shape)
plt.boxplot(data)

plt.xticks(list(range(1,11)), label)
plt.xlabel('Reduced Dimension')
plt.ylabel('CER for SYS')
plt.show()
