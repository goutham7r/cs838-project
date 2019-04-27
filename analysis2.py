import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

c=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=b.values[:, :-1]
print(b.shape)
print(b[1])


d ={}

for label in range(750):
	d[label] = b[b[:,0]==label]

a = []
for label in d:
	d[label] = np.mean(d[label], axis=0)
	a.append(d[label])
	# print(d[label].shape)

a = np.array(a)
# print(a[:5])


# plt.figure()
# ax = sns.heatmap(a[:80,7:30], linewidth=0.5, cmap="YlGnBu")
# # ax.set_title('Facial Hair')
# plt.show()

i = 3
c = c.values[:,i]
print(c.shape)
vals = np.unique(c)
print(vals)


indices = np.array([])
arr = np.arange(750)
freq = [0]
for val in vals:
	indices = np.append(indices, arr[c==val])
	freq.append(indices.shape[0])

# print(indices)

print(freq)
indices = indices.astype(int)
a = a[indices]

for i in range(1,len(freq)):
	plt.figure()
	ax = sns.heatmap(a[freq[i-1]:freq[i],7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title(str(vals[i-1]))
plt.show()

# indices.append()

