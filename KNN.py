import numpy as np
import pandas

def _mostFound(labels):
    words = []
    for i in range(len(labels)):
        if labels[i] not in words:
            words.append(labels[i])

    mostCounted = ''
    nMostCounted = None

    for i in range(len(words)):

        counted = labels.count(words[i])

        if nMostCounted == None:

            mostCounted = words[i]
            nMostCounted = counted

        elif nMostCounted < counted:

            mostCounted = words[i]
            nMostCounted = counted

        elif nMostCounted == counted:

            mostCounted = None

    return mostCounted


def _findNeighbors(newPoint, data, labels, k):

    dim = len(newPoint)
    neighbors = []
    neighborLabels = []

    for i in range(0, k):
        nearestNeighbor = None
        shortestDistance = None

        for i in range(0, len(data)):
            distance = 0
            for d in range(0, dim):

                dist = abs(newPoint[d] - data[i][d])
                distance += dist**2

            distance = np.sqrt(distance)

            if shortestDistance == None:

                shortestDistance = distance
                nearestNeighbor = i

            elif shortestDistance > distance:

                shortestDistance = distance
                nearestNeighbor = i
        

        neighbors.append(data[nearestNeighbor])
        neighborLabels.append(labels[nearestNeighbor])
        
        data.remove(data[nearestNeighbor])
        labels.remove(labels[nearestNeighbor])

    return neighborLabels

def knn(newPoint, data, labels, k):
    """
    Function that gets a dataframe as a list, a list of labels for the data, a new point with the same dimensions as the dataframe and the
    k neighbors that we want to evaluate in order to identify a label for the newPoint
    """
    #Convert dataframes to lists in order to simplify the work
    df = data
    while True:

        neighborLabels = _findNeighbors(newPoint, df, labels, k)
        label = _mostFound(neighborLabels)
        
        if label != None:
            break
        
        #If we don't get a nearest label for this point, it might be because we found two label candidates.
        #So we increase the value of k in order to get results
        k += 1

        if k >= len(df):
            break

    return label