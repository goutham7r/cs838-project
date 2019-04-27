import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels_train.out',delimiter=' ',header=None)
b=np.transpose(b)

black=[0.0 for i in range(33)]
asian=[0.0 for i in range(33)]
indian=[0.0 for i in range(33)]
arab=[0.0 for i in range(33)]
latino=[0.0 for i in range(33)]
caucasian=[0.0 for i in range(33)]
black_c=0
asian_c=0
indian_c=0
arab_c=0
latino_c=0
caucasian_c=0
unk_c=0

for j in range(59999):
	if(a['Race'][b[j][0]]==0):
		for i in range(33):
			black[i]+=b[j][i+1]
		black_c+=1
	elif(a['Race'][b[j][0]]==1):
		for i in range(33):
			asian[i]+=b[j][i+1]
		asian_c+=1
	elif(a['Race'][b[j][0]]==2):
		for i in range(33):
			indian[i]+=b[j][i+1]
		indian_c+=1
	elif(a['Race'][b[j][0]]==3):
		for i in range(33):
			arab[i]+=b[j][i+1]
		arab_c+=1
	elif(a['Race'][b[j][0]]==4):
		for i in range(33):
			latino[i]+=b[j][i+1]
		latino_c+=1
	elif(a['Race'][b[j][0]]==5):
		for i in range(33):
			caucasian[i]+=b[j][i+1]
		caucasian_c+=1
	else:
		unk_c+=1

# print(unk_c/10)

race=np.zeros([6,33])
race[0]=np.true_divide(black,black_c)
race[1]=np.true_divide(asian,asian_c)
race[2]=np.true_divide(indian,indian_c)
race[3]=np.true_divide(arab,arab_c)
race[4]=np.true_divide(latino,latino_c)
race[5]=np.true_divide(caucasian,caucasian_c)

ax = sns.heatmap(race[:,7:30], linewidth=0.5)
ax.set_title('Race')

plt.show()