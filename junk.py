import json, csv

a = json.load(open('Age_Gender.json'))

f = open('blah.csv', 'w')
writer = csv.writer(f, delimiter=',')

keys = [str(i) for i in range(750)]

c=0
for e in keys:
	# print(a[e])
	# break
	b = a[e]
	if 'Age' not in b:
		b['Age'] = "-1"
	if int(b['Age']) > 70:
		c+=1	
	writer.writerow([b['Age']])
print(c)
f.close() 

