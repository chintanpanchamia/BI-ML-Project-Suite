'''Project discussed with kdesai2 and ashah7'''

import csv
import random
import collections
import sys
import math

'''Make one dict for bids, and an array for bidders' budgets, when you get up'''

if len(sys.argv) != 2:
    print "python adwords.py <greedy | mssv | balance>"
    exit(1)
func = sys.argv[1]

bidderBudget = [-1]*100
bidderDict = collections.defaultdict(list)
with open('bidder_dataset.csv', 'rb') as bidderCSV:
	has_header = csv.Sniffer().has_header(bidderCSV.read(1024))
	bidderCSV.seek(0)
	bidderData = csv.reader(bidderCSV)
	if has_header:
		next(bidderData)
		ad_id = -1
	for row in bidderData:
		if int(row[0]) != ad_id:
			ad_id = int(row[0])
			bidderBudget[ad_id] = float(row[3])
			#bidderDict[row[1]].append((ad_id, int(row[2])))
		bidderDict[row[1]].append((ad_id, float(row[2])))

def greedy(myDict, myBudgets, queries):
	revenue = 0
	
	for q in queries:
		q = q.strip()
		
		#check budget - bid of advertisers for query q
		ad = [x if myBudgets[x[0]] - x[1] >= 0 else (-1, -1) for x in myDict[q]]
		
		#get Only ones above 0, select highest amongst them, add to revenue
		ad1 = [(-1,-1)] * len(ad)
		if ad == ad1:
			continue
		winner = max([x[1] for x in ad])
		
		revenue += winner
		
		myBudgets[ad[[i for i, v in enumerate(ad) if v[1] == winner][0]][0]] -= winner
	return revenue


def mssv(myDict, myBudgets, queries):
	revenue = 0

	for q in queries:
		q = q.strip()

		ad = [x if myBudgets[x[0]] - x[1] >= 0 else (-1, -1) for x in myDict[q]]
		ad1 = [(-1,-1)] * len(ad)
		if ad == ad1:
			continue
		psiX = ad[:]
		psiX = map(lambda x: (1 - math.exp(-1 * float(myBudgets[x[0]])/bidderBudget[x[0]])) * x[1], psiX)
		
		k = psiX.index(max(psiX))
		
		revenue += ad[k][1]

		myBudgets[ad[k][0]] -= ad[k][1]
	return revenue


def balance(myDict, myBudgets, queries):
	revenue = 0

	for q in queries:
		q = q.strip()
		ad = [x if myBudgets[x[0]] - x[1] >= 0 else (-1, -1) for x in myDict[q]]

		ad1 = [(-1, -1)] * len(ad)
		if ad == ad1:
			continue
		tempBudget = [myBudgets[x[0]] if x[0] != -1 else -1 for x in ad]
		winner = tempBudget.index(max(tempBudget))

		revenue += ad[winner][1]

		myBudgets[ad[winner][0]] -= ad[winner][1]
	return revenue



def core(func, incomingQueries):
	opt = sum(bidderBudget)
	total = 0
	with open(incomingQueries, 'rb') as f:
		queries = f.readlines()
	if func == 'greedy':
		for i in range(100):
			tempBudget = bidderBudget[:]
			total += greedy(bidderDict, tempBudget, queries)
			random.shuffle(queries)
		total = total/100
		print (total, total/opt)

	elif func == 'mssv':
		for i in range(100):
			tempBudget = bidderBudget[:]
			total += mssv(bidderDict, tempBudget, queries)
			random.shuffle(queries)
		total = total/100
		print (total, total/opt)

	elif func == 'balance':
		for i in range(100):
			tempBudget = bidderBudget[:]
			total += balance(bidderDict, tempBudget, queries)
			random.shuffle(queries)
		total = total/100
		print (total, total/opt)



core(func, 'queries.txt')