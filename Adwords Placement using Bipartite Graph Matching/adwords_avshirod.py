import sys
import csv
import math
import random
import collections

queryGraph = {}
advBudget = {}
# advertiserID_i, queries_q, bid_iq, budget_bi
# bidder_dataset.csv
OPT = 0.0
with open('bidder_dataset.csv', 'r') as csvfile:
  header = csv.Sniffer().has_header(csvfile.read(1024))
  csvfile.seek(0)
  bidderData = csv.reader(csvfile)
  if(header):
    next(bidderData)
  advertiserID_i = -1
  for row in bidderData:
    # print(', '.join(row))
    if int(row[0]) != advertiserID_i:
      # print(int(row[0]), row[3], advertiserID_i, row[1])
      advertiserID_i, queries_q, bid_iq = int(row[0]), row[1].strip(), float(row[2])
      if not (row[3] == ''):
        budget_bi = float(row[3])
        advBudget[advertiserID_i] = budget_bi
        OPT += budget_bi
    else: queries_q, bid_iq = row[1], float(row[2])
    #if (advertiserID_i not in advBudget):
      #advBudget[advertiserID_i] = budget_bi
    if (queries_q in queryGraph):
      queryGraph[queries_q].append((advertiserID_i, bid_iq))
    else: queryGraph[queries_q] = [(advertiserID_i, bid_iq)]

advBudgetBackup = dict(advBudget)
# print(OPT)
# queries.txt
# Calculate revenue and competitive ratio (min(ALG/OPT)) (ALG - Mean std revenue over 100 random repetitions of queries, OPT - Optimal matching)

with open('queries.txt', 'r') as qfile:
  q = qfile.readlines()

def greedy():
  revenue = 0
  for query in q:
    query = query.strip()
    if(not(exhaustBudget(query))):
      revenue += matchNeighbourGreedy(query)
  return revenue

def msvv():
  revenue = 0
  advBudgetCopy = dict(advBudget)
  for query in q:
    query = query.strip()
    if(not(exhaustBudget(query))):
      revenue += matchNeighbourMSVV(query, advBudgetCopy)
  return revenue

def balance():
  revenue = 0
  for query in q:
    query = query.strip()
    if(not(exhaustBudget(query))):
      revenue += matchNeighbourBalance(query)
  return revenue

def exhaustBudget(query):
  qneighbours = queryGraph[query]
  adv, bids = map(list, zip(*qneighbours))
  budget = [advBudget[a] for a in adv]
  # print(budget)
  if sum(budget) == 0: 
    return True
    # print("Budget Exhausted")
  return False

def matchNeighbourGreedy(query):
  qneighbours = queryGraph[query]
  adv, bids = map(list, zip(*qneighbours))
  maxBid = max(bids)
  maxBidAdvID = adv[bids.index(maxBid)]
  advBudget[maxBidAdvID] -= maxBid
  # print(maxBidAdvID, maxBid)
  # print(advBudget)
  return maxBid

def matchNeighbourMSVV(query, advBudgetCopy):
  qneighbours = queryGraph[query]
  adv, bids = map(list, zip(*qneighbours))
  spentBudget = [advBudget[a] for a in adv]
  originalBudget = [advBudgetCopy[a] for a in adv]
  fracBudgetSpent = [((original-spent)/original) for original,spent in zip(originalBudget, spentBudget)]
  chiBudget = [(1 - math.exp(xi - 1)) for xi in fracBudgetSpent]
  msvvValues = [(bi * chiValue) for bi,chiValue in zip(bids,chiBudget)]
  maxMSVVValue = max(msvvValues)
  maxMSVVValueAdvID = adv[msvvValues.index(maxMSVVValue)]
  advBudget[maxMSVVValueAdvID] -= bids[msvvValues.index(maxMSVVValue)]
  return bids[msvvValues.index(maxMSVVValue)]

def matchNeighbourBalance(query):
  qneighbours = queryGraph[query]
  adv, bids = map(list, zip(*qneighbours))
  budget = [advBudget[a] for a in adv]
  maxUnspentBudget = max(budget)
  maxUnspentBudgetAdvID = adv[budget.index(maxUnspentBudget)]
  advBudget[maxUnspentBudgetAdvID] -= bids[budget.index(maxUnspentBudget)]
  return bids[budget.index(maxUnspentBudget)]
  # for a in adv:
  #   budget.append(advBudget[a])

total = 0
random.seed(0)
if(sys.argv[1] == "greedy"):
  for i in range(100):
    advBudget = dict(advBudgetBackup)
    total += greedy()
    random.shuffle(q)
  total /= 100
  print(total, total/OPT)
elif (sys.argv[1] == "msvv"):
  for i in range(100):
    advBudget = dict(advBudgetBackup)
    total += msvv()
    random.shuffle(q)
  total /= 100
  print(total, total/OPT)
elif (sys.argv[1] == "balance"):
  for i in range(100):
    advBudget = dict(advBudgetBackup)
    total += balance()
    random.shuffle(q)
  total /= 100
  print(total, total/OPT)
else:
  print("Please follow the syntax - 'python adwords.py greedy' or 'python adwords.py balance' or 'python adwords.py msvv'")
  exit(1)