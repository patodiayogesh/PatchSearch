import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import json

a = []
f = open("edit_dasnD", "r")

for line in f:
    l = line.strip('\n')
    a.append(float(line))
f.close()

print(len(a))
sns.displot(a)
plt.show()
plt.savefig('test')

b = list(np.random.rand(100))
with open('pelin','w') as f:
    for elem in b:
        f.write(str(elem) + "\n")
f.close()

with open('pelin','r') as f:
    b.append(f.readline().strip('\n'))

print(b)
