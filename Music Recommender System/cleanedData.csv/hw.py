import csv
import numpy as np
'''import matplotlib.pyplot as plt

def parser(var):
	if var == '':
		var = 0
	else:
		var = var.replace(',','')
	return float(var)'''

reader=csv.reader(open("part-0000.csv","r"))

data_x = []

for line in reader:
	data_x.append(line)
	

x = np.array(data_x)

#x.dump("MusicData.dat")

test = x[:10000]
train = x[10000:40000]
validation = x[40000:]

test.dump("MusicTest.dat")
train.dump("MusicTrain.dat")
validation.dump("MusicValidation.dat")
#print len(test), len(train), len(validation)

#y = np.array(data_y)

'''
q_Facebook = np.array(data_Q5)
q_Email_Shares = np.array(data_Q9)


def answer_to_Q2():
	count = 0
	for item in y:
		if item >= 1:
			count += 1
	print('Q2 ' + str(count))

def answer_to_Q3():
	count = 0
	for item in y:
		if item >= 100:
			count += 1
	print('Q3 ' + str(count))

def answer_to_Q4():
	count = 0
	for item in y:
		if item >= 1000:
			count += 1
	print('Q4 ' + str(count))

def answer_to_Q5():
	count = 0
	for item in q_Facebook:
		if item >= 1:
			count += 1
	print('Q5 ' + str(count))

def answer_to_Q6():
	count = 0
	for item in q_Facebook:
		if item >= 100:
			count += 1
	print('Q6 ' + str(count))

def answer_to_Q7():
	count = 0
	for item in q_Facebook:
		if item >= 1000:
			count += 1
	print('Q7 ' + str(count))

def answer_to_Q8():
	count = 0
	for item in q_Facebook:
		if item >= 10000:
			count += 1
	print('Q8 ' + str(count))

def answer_to_Q9():
	count = 0
	for item in q_Email_Shares:
		if item >= 1:
			count += 1
	print('Q9 ' + str(count))

answer_to_Q2()
answer_to_Q3()
answer_to_Q4()
answer_to_Q5()
answer_to_Q6()
answer_to_Q7()
answer_to_Q8()
answer_to_Q9()


plt.scatter(x, y)
plt.xlim([0, 2000000])
plt.ylim([0, 5000])
plt.show()


'''









'''def connector(a, b):
	indexA = len(a) - 1
	indexB = len(b) - 1
	output = []
	while indexA >= 0 and indexB >= 0:
		if a[indexA] > b[indexB]:
			output = [a[indexA]] + output
			indexA -= 1
		elif a[indexA] < b[indexB]:
			output = [b[indexB]] + output
			indexB -= 1
		else:
			output = [a[indexA]] + output
			output = [b[indexB]] + output
			indexA -= 1
			indexB -= 1

	if indexA < 0:
		output = b[:indexB + 1] + output
	else:
		output = a[:indexA + 1] + output

	return output

print connector([1,4,5,6,9],[3,6,7,10])



'''
























