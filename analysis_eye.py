import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

a=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=np.transpose(b)

black=[0.0 for i in range(33)]
brown=[0.0 for i in range(33)]
blue=[0.0 for i in range(33)]
blue_c=0
brown_c=0
black_c=0
unk_c=0

for j in range(7499):
	if(a['Eye'][b[j][0]]==2):
		for i in range(33):
			blue[i]+=b[j][i+1]
		blue_c+=1
	elif(a['Eye'][b[j][0]]==1):
		for i in range(33):
			brown[i]+=b[j][i+1]
		brown_c+=1
	elif(a['Eye'][b[j][0]]==0):
		for i in range(33):
			black[i]+=b[j][i+1]
		black_c+=1
	else:
		unk_c+=1

# print(unk_c/10)

eye=np.zeros([3,33])
eye[2]=np.true_divide(blue,blue_c)
eye[1]=np.true_divide(brown,brown_c)
eye[0]=np.true_divide(black,black_c)

ax = sns.heatmap(eye[:,7:30], linewidth=0.5)
ax.set_title('Eye Colour')

plt.show()