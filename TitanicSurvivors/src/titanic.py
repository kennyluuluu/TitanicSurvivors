"""
Description : Titanic
"""

## IMPORTANT: Use only the provided packages!

## SOME SYNTAX HERE.   
## I will use the "@" symbols to refer to some variables and functions. 
## For example, for the 3 lines of code below
## x = 2
## y = x * 2 
## f(y)
## I will use @x and @y to refer to variable x and y, and @f to refer to function f

import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics


######################################################################
# classes
######################################################################

class Classifier(object) :

    ## THIS IS SOME GENERIC CLASS, YOU DON'T NEED TO DO ANYTHING HERE. 

    """
    Classifier interface.
    """

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) : ## INHERITS FROM THE @CLASSIFIER

    def __init__(self) :
        """
        A classifier that always predicts the majority class.

        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None

    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self

    def predict(self, X) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")

        # n,d = X.shape ## get number of sample and dimension
        y = [self.prediction_] * X.shape[0]
        return y


class RandomClassifier(Classifier) :

    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.

        Attributes
        --------------------
            probabilities_ -- an array specifying probability to survive vs. not 
        """
        self.probabilities_ = None ## should have length 2 once you call @fit

    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes

        Returns
        --------------------
            self -- an instance of self
        """

        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        # in simpler wordings, find the probability of survival vs. not
        
        survivors = 0
        for val in y:
            if val == 1:
                survivors += 1
        total = len(y)
        casualties = total - survivors
        self.probabilities_ = (survivors/total, casualties/total)
        ### ========== TODO : END ========== ###

        return self

    def predict(self, X, seed=1234) :
        """
        Predict class values.

        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed

        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)

        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (check the arguments of np.random.choice) to randomly pick a value based on the given probability array @self.probabilities_
        y = []
        for i in range(X.shape[0]):
            r = np.random.random()
            if r < self.probabilities_[0]:
                y.append(1)
            else:
                y.append(0)
        ### ========== TODO : END ========== ###

        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')

    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """

    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))

    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'

    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.

    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials

    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """

    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use @train_test_split to split the data into train/test set 
    # xtrain, xtest, ytrain, ytest = train_test_split (X,y, test_size = test_size, random_state = i)
    # now you can call the @clf.fit (xtrain, ytrain) and then do prediction
    train_error = 0 ## average error over all the @ntrials
    test_error = 0
    train_scores = []; test_scores = []; ## tracking the error for each of the @ntrials, these array should have length 100 once you're done. 
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train, y_train)
        
        train_pred = clf.predict(X_train)
        train_err = 1 - metrics.accuracy_score(y_train, train_pred, normalize=True)
        train_scores.append(train_err)
        
        test_pred = clf.predict(X_test)
        test_err = 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)
        test_scores.append(test_err)
    train_error = sum(train_scores)/ntrials
    test_error = sum(test_scores)/ntrials
    ### ========== TODO : END ========== ###

    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features



    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    MVclf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    MVclf.fit(X, y)                  # fit training data using the classifier
    MV_y_pred = MVclf.predict(X)        # take the classifier and run it on the training data
    MV_train_error = 1 - metrics.accuracy_score(y, MV_y_pred, normalize=True)
    print('\t-- training error: %.3f' % MV_train_error)



    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random Classifier...')
    random_clf = RandomClassifier()
    random_clf.fit(X, y)
    random_y_pred = random_clf.predict(X)
    random_train_error = 1 - metrics.accuracy_score(y, random_y_pred, normalize=True)
    print('\t-- training error: %.3f' % random_train_error)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain
    print('Classifying using Decision Tree...')
    # call the function @DecisionTreeClassifier
    DTclf = DecisionTreeClassifier(criterion='entropy')
    DTclf.fit(X, y)
    DT_y_pred = DTclf.predict(X)
    DT_train_error = 1 - metrics.accuracy_score(y, DT_y_pred, normalize=True)
    print('\t-- training error: %.3f' % DT_train_error)
    ### ========== TODO : END ========== ###


    # note: uncomment out the following lines to output the Decision Tree graph
    
    # save the classifier -- requires GraphViz and pydot
    # import io, pydot
    # from sklearn import tree
    # dot_data = io.StringIO()
    # tree.export_graphviz(DTclf, out_file=dot_data,
                         # feature_names=Xnames)
    # (graph, ) = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("dtree.pdf")
    



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors
    print('Classifying using k-Nearest Neighbors for k = 3...')
    # call the function @KNeighborsClassifier
    kNN3clf = KNeighborsClassifier(n_neighbors=3)
    kNN3clf.fit(X, y)
    kNN3_y_pred = kNN3clf.predict(X)
    kNN3_train_error = 1 - metrics.accuracy_score(y, kNN3_y_pred, normalize=True)
    print('\t-- training error: %.3f' % kNN3_train_error)
    
    print('Classifying using k-Nearest Neighbors for k = 5...')
    kNN5clf = KNeighborsClassifier(n_neighbors=5)
    kNN5clf.fit(X, y)
    kNN5_y_pred = kNN5clf.predict(X)
    kNN5_train_error = 1 - metrics.accuracy_score(y, kNN5_y_pred, normalize=True)
    print('\t-- training error: %.3f' % kNN5_train_error)
    
    print('Classifying using k-Nearest Neighbors for k = 7...')
    kNN7clf = KNeighborsClassifier(n_neighbors=7)
    kNN7clf.fit(X, y)
    kNN7_y_pred = kNN7clf.predict(X)
    kNN7_train_error = 1 - metrics.accuracy_score(y, kNN7_y_pred, normalize=True)
    print('\t-- training error: %.3f' % kNN7_train_error)
    ### ========== TODO : END ========== ###
    


    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    # call your function @error
    print('Performing cross-validation on MajorityVoteClassifier...')
    MV_train_error, MV_test_error = error(MajorityVoteClassifier(), X, y)
    print('\t-- training error: %.3f' % MV_train_error)
    print('\t-- testing error: %.3f' % MV_test_error)
    
    print('Performing cross-validation on RandomClassifier...')
    Random_train_error, Random_test_error = error(RandomClassifier(), X, y)
    print('\t-- training error: %.3f' % Random_train_error)
    print('\t-- testing error: %.3f' % Random_test_error)
    
    print('Performing cross-validation on DecisionTreeClassifier...')
    DT_train_error, DT_test_error = error(DecisionTreeClassifier(criterion='entropy'), X, y)
    print('\t-- training error: %.3f' % DT_train_error)
    print('\t-- testing error: %.3f' % DT_test_error)
    
    print('Performing cross-validation on KNeighborsClassifier...')
    KNN_train_error, KNN_test_error = error(KNeighborsClassifier(n_neighbors=5), X, y)
    print('\t-- training error: %.3f' % KNN_train_error)
    print('\t-- testing error: %.3f' % KNN_test_error)
    ### ========== TODO : END ========== ###


    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    # hint: use the function @cross_val_score
    k = list(range(1,50,2))
    cv_score = [] ## track accuracy for each value of $k, should have length 25 once you're done
    for i in k:
        val_errors = cross_val_score(KNeighborsClassifier(n_neighbors=i), X, y, cv=10)
        avg = sum(val_errors)/10
        cv_score.append(avg)
        
    import matplotlib
    import matplotlib.pyplot as plt
  
    plt.plot(k, cv_score)
    plt.title('K-Neighbors vs. Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.savefig("4f.png")
    plt.clf()
    ### ========== TODO : END ========== ###


    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    train_errors = []
    test_errors = []
    k = range(1,21)
    for i in k:
        DT_train_error, DT_test_error = error(DecisionTreeClassifier(criterion='entropy', max_depth=i), X, y)
        train_errors.append(DT_train_error)
        test_errors.append(DT_test_error)
        
    plt.plot(k, train_errors, label='train')
    plt.plot(k, test_errors, label='test')
    plt.title('Max Depth Effect on Decision Tree Classifier')
    plt.ylabel('Average Error')
    plt.xlabel('Decision Tree Max Depth')
    plt.legend()
    plt.savefig("4g.png")
    plt.clf()
    ### ========== TODO : END ========== ###


    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)
    DT_train_errors = []
    DT_test_errors = []
    KNN_train_errors = []
    KNN_test_errors = []
    k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for fraction in k:
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for i in range(100):
            xi_train, _,yi_train , _= train_test_split(X_train, y_train, test_size=fraction)
            
            DTclf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
            DTclf.fit(xi_train, yi_train)
            
            DT_train_error = 1 - metrics.accuracy_score(yi_train, DTclf.predict(xi_train), normalize=True)
            sum1 += DT_train_error
            
            DT_test_error = 1 - metrics.accuracy_score(y_test, DTclf.predict(X_test), normalize=True)
            sum2 += DT_test_error
            
            
            KNNclf = KNeighborsClassifier(n_neighbors=7)
            KNNclf.fit(xi_train, yi_train)
            
            KNN_train_error = 1 - metrics.accuracy_score(yi_train, KNNclf.predict(xi_train), normalize=True)
            sum3 += KNN_train_error
            
            KNN_test_error = 1 - metrics.accuracy_score(y_test, KNNclf.predict(X_test), normalize=True)
            sum4 += KNN_test_error
        
        DT_train_errors.append(sum1/100)
        DT_test_errors.append(sum2/100)
        KNN_train_errors.append(sum3/100)
        KNN_test_errors.append(sum4/100)
        

    plt.plot(k, DT_train_errors, label='DT-train')
    plt.plot(k, DT_test_errors, label='DT-test')
    plt.plot(k, KNN_train_errors, label='KNN-train')
    plt.plot(k, KNN_test_errors, label='KNN-test')
    plt.title('Learning Curve')
    plt.ylabel('Error')
    plt.xlabel('Percentage Trained')
    plt.legend()
    plt.savefig("4h.png")
    plt.clf()
    ### ========== TODO : END ========== ###


    print('Done')


if __name__ == "__main__":
    main()
