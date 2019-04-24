import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=np.transpose(b)

specs=[0.0 for i in range(33)]
no_specs=[0.0 for i in range(33)]
specs_c=0
no_specs_c=0
unk_c=0

for j in range(7499):
	if(a['Specs'][b[j][0]]==0):
		for i in range(33):
			specs[i]+=b[j][i+1]
		specs_c+=1
	elif(a['Specs'][b[j][0]]==1):
		for i in range(33):
			no_specs[i]+=b[j][i+1]
		no_specs_c+=1
	else:
		unk_c+=1

#print(unk_c/10)

spec=np.zeros([2,33])
spec[0]=np.true_divide(specs,specs_c)
spec[1]=np.true_divide(no_specs,no_specs_c)

ax = sns.heatmap(spec[:,7:30], linewidth=0.5)
ax.set_title('Spectacles')
plt.show()