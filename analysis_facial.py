import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=np.transpose(b)

none=[0.0 for i in range(33)]
beard=[0.0 for i in range(33)]
moustache=[0.0 for i in range(33)]
both=[0.0 for i in range(33)]
none_c=0
beard_c=0
moustache_c=0
both_c=0
unk_c=0

for j in range(7499):
	if(a['Facial_Hair'][b[j][0]]==0):
		for i in range(33):
			none[i]+=b[j][i+1]
		none_c+=1
	elif(a['Facial_Hair'][b[j][0]]==1):
		for i in range(33):
			beard[i]+=b[j][i+1]
		beard_c+=1
	elif(a['Facial_Hair'][b[j][0]]==2):
		for i in range(33):
			moustache[i]+=b[j][i+1]
		moustache_c+=1
	elif(a['Facial_Hair'][b[j][0]]==3):
		for i in range(33):
			both[i]+=b[j][i+1]
		both_c+=1
	else:
		unk_c+=1

# print(unk_c/10)

facial=np.zeros([4,33])
facial[0]=np.true_divide(none,none_c)
facial[1]=np.true_divide(beard,beard_c)
facial[2]=np.true_divide(moustache,moustache_c)
facial[3]=np.true_divide(both,both_c)

ax = sns.heatmap(facial[:,7:30], linewidth=0.5)
ax.set_title('Facial Hair')
plt.show()