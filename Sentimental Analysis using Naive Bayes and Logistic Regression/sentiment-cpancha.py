#This project was discussed with Akshat Shah (ashah7), and Kevin Desai (kdesai2)

import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) #To filter out the DeprecationWarning obtained when while executing the code
random.seed(0)

from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    #print len(train_pos), len(train_neg)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg

#def remover(the_list, v):
#    return [value for value in the_list if value != v]

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = list(set(nltk.corpus.stopwords.words('english')))
    
    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE
    finalPos = []
    finalNeg = []

    finalPos = [item for sublist in train_pos for item in set(sublist) if item not in stopwords]
    finalNeg = [item for sublist in train_neg for item in set(sublist) if item not in stopwords]

    #print len(finalPos), len(finalNeg)

    posCount = len(train_pos)/100
    negCount = len(train_neg)/100
    
    everythingPosIGot = {}
    everythingNegIGot = {}

    everythingPosIGot = collections.Counter(finalPos)
    everythingNegIGot = collections.Counter(finalNeg)
    totalWords = set(finalPos + finalNeg)

    tempFeatureWords = [word for word in totalWords if everythingPosIGot[word] >= posCount or everythingNegIGot[word] >= negCount]    
    
    featureWords = [word for word in tempFeatureWords if everythingPosIGot[word] >= 2*everythingNegIGot[word] or everythingNegIGot[word] >= 2*everythingPosIGot[word]]
    
    #print len(featureWords)
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []
    for line in train_pos:
        output = []
        for word in featureWords:
            if word in line:
                output += [1]
            else:
                output += [0]
        train_pos_vec.append(output)

    for line in train_neg:
        output = []
        for word in featureWords:
            if word in line:
                output += [1]
            else:
                output += [0]
        train_neg_vec.append(output)

    for line in test_pos:
        output = []
        for word in featureWords:
            if word in line:
                output += [1]
            else:
                output += [0]
        test_pos_vec.append(output)

    for line in test_neg:
        output = []
        for word in featureWords:
            if word in line:
                output += [1]
            else:
                output += [0]
        test_neg_vec.append(output)


    # Return the four feature vectors
    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE

    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    for line in range(len(train_pos)):
        label = 'TRAIN_POS_' + str(line)
        labeled_train_pos.append(LabeledSentence(words=train_pos[line], tags=[label]))

    for line in range(len(train_neg)):
        label = 'TRAIN_NEG_' + str(line)
        labeled_train_neg.append(LabeledSentence(words=train_neg[line], tags=[label]))

    for line in range(len(test_pos)):
        label = 'TEST_POS_' + str(line)
        labeled_test_pos.append(LabeledSentence(words=test_pos[line], tags=[label]))

    for line in range(len(test_neg)):
        label = 'TEST_NEG_' + str(line)
        labeled_test_neg.append(LabeledSentence(words=test_neg[line], tags=[label]))

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE


    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    for line in range(len(train_pos)):
        label = 'TRAIN_POS_' + str(line)
        train_pos_vec.append(model.docvecs[label])

    for line in range(len(train_neg)):
        label = 'TRAIN_NEG_' + str(line)
        train_neg_vec.append(model.docvecs[label])

    for line in range(len(test_pos)):
        label = 'TEST_POS_' + str(line)
        test_pos_vec.append(model.docvecs[label])

    for line in range(len(test_neg)):
        label = 'TEST_NEG_' + str(line)
        test_neg_vec.append(model.docvecs[label])
    
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    X = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1, binarize=None).fit(X, Y)
    lr_model = sklearn.linear_model.LogisticRegression().fit(X, Y)
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    X = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.GaussianNB().fit(X, Y)
    lr_model = sklearn.linear_model.LogisticRegression().fit(X, Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    
    truePositives = 0.0
    falsePositives = 0.0
    trueNegatives = 0.0
    falseNegatives = 0.0
    
    for x in test_pos_vec:
        if model.predict(x) == ['pos']:
            truePositives = truePositives + 1
        else:
            falseNegatives = falseNegatives + 1

    for x in test_neg_vec:
        if model.predict(x) == ['neg']:
            trueNegatives = trueNegatives + 1
        else:
            falsePositives = falsePositives + 1

    accuracy = ((truePositives + trueNegatives)/(truePositives + trueNegatives + falsePositives + falseNegatives))
    
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (truePositives, falseNegatives)
        print "neg\t\t%d\t%d" % (falsePositives, trueNegatives)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
