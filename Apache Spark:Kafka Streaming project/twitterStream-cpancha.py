from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt

conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
ssc.checkpoint("checkpoint")

#This project was discussed with Kevin Desai (kdesai2)

def main():
    
    
    #print pwords
    #print '\n\n\n\n\n\n\n\n\n\n'
    #print nwords
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    ncount = []
    pcount = []
    for i in range(0, len(counts)):
        j = counts[i]
        if j != []:
            pcount.append(j[0][1])
            ncount.append(j[1][1])
    
    plt.plot(pcount, label="positive", marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.plot(ncount, label="negative", marker='o')
    
    plt.legend(loc="upper left", bbox_to_anchor=[0, 1], ncol=1, fancybox=True)
    
    plt.xlim=[0,11]
    plt.show()



def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    return sc.textFile(filename).collect()

pwords = load_wordlist("positive.txt")
nwords = load_wordlist("negative.txt")

def classifier(word):
    if word in pwords:
        return 'positive'
    elif word in nwords:
        return 'negative'
    else:
        return 'none'

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))
    #print "HELOKOJOJEORUBEORUBOUBEROUBNOUONEROJOEJRNOJENROJENFOJEFOEJFNOEFUNOEUFN"
    #tweets.pprint()
    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    words = tweets.flatMap(lambda line: line.split(" "))
    pairs = words.map(classifier).map(lambda word: (word, 1)).filter(lambda x: x[0] != 'none').reduceByKey(lambda a,b: a+b)
    runningCounts = pairs.updateStateByKey(updateFunction)
    runningCounts.pprint()
    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    pairs.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)
    #print counts
    return counts

def updateFunction(newValues, runningCount):
    if runningCount is None:
        runningCount = 0
    return sum(newValues, runningCount)

if __name__=="__main__":
    main()
