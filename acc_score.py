from sklearn import metrics
import numpy as np

acc = []

predicted = [0.29, 0.86, 0.11, 0.28] #test data


for probability_threshold in np.arange(0.4, 0.8, 0.02):
    # metrics is a dictionary of the different metric values i.e. { "accuracy": 0.6, "f1": 0.26}
    for i in range(0, 3):
        if predicted[i] > probability_threshold:
            print(i)
   # accuracy = metrics.accuracy_score(targets, outputs.argmax(axis=1)) #y_true, ypred
    #print(accuracy)

#y_pred = [0, 2, 1, 3]

#y_pred = [[0], [0, 1]]
#y_true = [[0, 1], [0, 2]]