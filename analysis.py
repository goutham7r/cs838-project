import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ndprint(a, format_string ='{0:.2f}'):
    print([(i,float(format_string.format(v,i))) for i,v in enumerate(a)])


a=pd.read_csv('manual_annotations.csv',delimiter=',')
# b=pd.read_csv('extract_labels.out',delimiter=' ',header=None)
b=pd.read_csv('aig_all_40_train.out',delimiter=' ',header=None)


b=b.values[:,:-1]
# b=np.transpose(b)

print(b.shape)
m = np.mean(b, axis=0)
ndprint(m)

num = b.shape[0]


def gender():
	print("Gender Analysis")
	male=[0.0 for i in range(33)]
	female=[0.0 for i in range(33)]
	male_c=0
	female_c=0
	unk_c=0

	for j in range(num):
		if(a['Gender'][b[j][0]]==0):
			for i in range(33):
				male[i]+=b[j][i+1]
			male_c+=1
		elif(a['Gender'][b[j][0]]==1):
			for i in range(33):
				female[i]+=b[j][i+1]
			female_c+=1
		else:
			unk_c+=1

	#print(unk_c/10)
	print(male_c, female_c, unk_c)

	gender=np.zeros([2,33])
	gender[0]=np.true_divide(male,male_c)
	gender[1]=np.true_divide(female,female_c)

	plt.figure()
	ax = sns.heatmap(gender[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Gender')
	# plt.show()


def age():
	print("Age Analysis")
	# -1, <25, 25-35, 35-45, 45-55, 55+
	unk = [0.0 for i in range(33)]
	unk_c = 0

	age_lt_25 = [0.0 for i in range(33)]
	age_lt_25_c = 0

	age_25_35 = [0.0 for i in range(33)]
	age_25_35_c = 0

	age_35_45 = [0.0 for i in range(33)]
	age_35_45_c = 0

	age_45_55 = [0.0 for i in range(33)]
	age_45_55_c = 0

	age_gt_55 = [0.0 for i in range(33)]
	age_gt_55_c = 0

	for j in range(num):
		age = a['Age'][b[j][0]]

		if age == -1:
			for i in range(33):
				unk[i]+=b[j][i+1]
			unk_c+=1
		elif age<25:
			for i in range(33):
				age_lt_25[i]+=b[j][i+1]
			age_lt_25_c+=1
		elif age<35:
			for i in range(33):
				age_25_35[i]+=b[j][i+1]
			age_25_35_c+=1
		elif age<45:
			for i in range(33):
				age_35_45[i]+=b[j][i+1]
			age_35_45_c+=1
		elif age<55:
			for i in range(33):
				age_45_55[i]+=b[j][i+1]
			age_45_55_c+=1
		else:
			for i in range(33):
				age_gt_55[i]+=b[j][i+1]
			age_gt_55_c+=1
	print(unk_c, age_lt_25_c, age_25_35_c, age_35_45_c, age_45_55_c, age_gt_55_c)

	age=np.zeros([6,33])
	age[0]=np.true_divide(unk,unk_c)
	age[1]=np.true_divide(age_lt_25,age_lt_25_c)
	age[2]=np.true_divide(age_25_35,age_25_35_c)
	age[3]=np.true_divide(age_35_45,age_35_45_c)
	age[4]=np.true_divide(age_45_55,age_45_55_c)
	age[5]=np.true_divide(age_gt_55,age_gt_55_c)

	plt.figure()
	ax = sns.heatmap(age[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Age')




def facial():
	print("Facial Analysis")
	none=[0.0 for i in range(33)]
	beard=[0.0 for i in range(33)]
	moustache=[0.0 for i in range(33)]
	both=[0.0 for i in range(33)]
	none_c=0
	beard_c=0
	moustache_c=0
	both_c=0
	unk_c=0

	for j in range(num):
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
	print(none_c, beard_c, moustache_c, both_c, unk_c)

	facial=np.zeros([4,33])
	facial[0]=np.true_divide(none,none_c)
	facial[1]=np.true_divide(beard,beard_c)
	facial[2]=np.true_divide(moustache,moustache_c)
	facial[3]=np.true_divide(both,both_c)

	plt.figure()
	ax = sns.heatmap(facial[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Facial Hair')
	# plt.show()



def eye():
	print("Eye Analysis")
	black=[0.0 for i in range(33)]
	brown=[0.0 for i in range(33)]
	blue=[0.0 for i in range(33)]
	blue_c=0
	brown_c=0
	black_c=0
	unk_c=0

	for j in range(num):
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
	print(black_c, brown_c, blue_c, unk_c)

	eye=np.zeros([3,33])
	eye[2]=np.true_divide(blue,blue_c)
	eye[1]=np.true_divide(brown,brown_c)
	eye[0]=np.true_divide(black,black_c)

	plt.figure()
	ax = sns.heatmap(eye[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Eye Colour')
	# plt.show()


def race():
	print("Race Analysis")
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

	for j in range(num):
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
	print(black_c, asian_c, indian_c, arab_c, latino_c, caucasian_c, unk_c)

	race=np.zeros([6,33])
	race[0]=np.true_divide(black,black_c)
	race[1]=np.true_divide(asian,asian_c)
	race[2]=np.true_divide(indian,indian_c)
	race[3]=np.true_divide(arab,arab_c)
	race[4]=np.true_divide(latino,latino_c)
	race[5]=np.true_divide(caucasian,caucasian_c)

	plt.figure()
	ax = sns.heatmap(race[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Race')

	# plt.show()


def skin():
	print("Skin Analysis")
	white=[0.0 for i in range(33)]
	brown=[0.0 for i in range(33)]
	black=[0.0 for i in range(33)]
	white_c=0
	brown_c=0
	black_c=0
	unk_c=0

	for j in range(num):
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

	# print(unk_c/10)
	print(black_c, brown_c, white_c, unk_c)

	skin=np.zeros([3,33])
	skin[2]=np.true_divide(white,white_c)
	skin[1]=np.true_divide(brown,brown_c)
	skin[0]=np.true_divide(black,black_c)

	plt.figure()
	ax = sns.heatmap(skin[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Skin Color')

	# plt.show()

def specs():
	print("Specs Analysis")
	specs=[0.0 for i in range(33)]
	no_specs=[0.0 for i in range(33)]
	specs_c=0
	no_specs_c=0
	unk_c=0

	for j in range(num):
		if(a['Specs'][b[j][0]]==1):
			for i in range(33):
				specs[i]+=b[j][i+1]
			specs_c+=1
		elif(a['Specs'][b[j][0]]==0):
			for i in range(33):
				no_specs[i]+=b[j][i+1]
			no_specs_c+=1
		else:
			unk_c+=1

	#print(unk_c/10)
	print(no_specs_c, specs_c, unk_c)

	spec=np.zeros([2,33])
	spec[1]=np.true_divide(specs,specs_c)
	spec[0]=np.true_divide(no_specs,no_specs_c)

	plt.figure()
	ax = sns.heatmap(spec[:,7:30], linewidth=0.5, cmap="YlGnBu")
	ax.set_title('Spectacles')
	# plt.show()

def main():
	facial()
	# specs()
	# eye()
	# race()
	# skin()
	gender()
	age()
	plt.show()



if __name__=="__main__":
	main()