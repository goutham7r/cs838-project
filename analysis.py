import pandas as pd
import csv
import numpy as np

a=pd.read_csv('manual_annotations.csv',delimiter=',')
b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=np.transpose(b)

white=[0.0 for i in range(33)]
brown=[0.0 for i in range(33)]
black=[0.0 for i in range(33)]
white_c=0
brown_c=0
black_c=0
unk_c=0

for j in range(7499):
	if(a['Skin'][b[j][0]]==2):
		for i in range(33):
			white[i]+=b[j][i+1]
		white_c+=1
	elif(a['Skin'][b[j][0]]==1):
		for i in range(33):
			brown[i]+=b[j][i+1]
		brown_c+=1
	elif(a['Skin'][b[j][0]]==0):
		for i in range(33):
			black[i]+=b[j][i+1]
		black_c+=1
	else:
		unk_c+=1

print(np.true_divide(white,white_c))
print(np.true_divide(brown,brown_c))
print(np.true_divide(black,black_c))
print(unk_c/10)
#print(b['g1'][3])