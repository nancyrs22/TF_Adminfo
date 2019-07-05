import numpy as np
import random

def randomCenters(dim, k):
    centers = []
    for _ in range(k):
        center = []

        for _ in range(dim):
            rand = random.random()
            center.append(rand)

        centers.append(center)
    return centers


def clustering(data, centers, dim, first_cluster):

    for point in data:

        nearestCenter = 0
        nearestCenterDist = None

        for i in range(0, len(centers)):
            distance = 0

            for d in range(0, dim):
                dist = abs(point[d] - centers[i][d])
                distance += dist**2

            distance = np.sqrt(distance)

            if nearestCenterDist == None:
                nearestCenterDist = distance
                nearestCenter = i

            elif nearestCenterDist > distance:
                nearestCenterDist = distance
                nearestCenter = i

        if first_cluster:
            point.append(nearestCenter)
        else:
            point[dim] = nearestCenter

    return data


def fixCenters(data, centers, dim):

    newCenters = []

    for i in range(len(centers)):

        center = []
        totalPoints = 0
        totalPointsSum = []
        
        for point in data:

            if point[dim] == i:
                totalPoints += 1

                for pos in range(0,dim):
                    if pos < len(totalPointsSum):
                        totalPointsSum[pos] += point[pos]
                    else:
                        totalPointsSum.append(point[pos])

        if len(totalPointsSum) != 0:

            for pos in range(0,dim):
                
                center.append(totalPointsSum[pos]/totalPoints)

            newCenters.append(center)
        else: 
            newCenters.append(centers[i])
            
    return newCenters


def k_meansInit(data, k, t):

    dim = len(data[0])
    centers = randomCenters(dim,k)
    
    clusteredData = clustering(data, centers, dim, True)

    for _ in range(t):
        centers = fixCenters(clusteredData, centers, dim)
        clusteredData = clustering(data, centers, dim, False)
    
    return centers, clusteredData


def k_meansPredict(point, centers):

    dim = len(point)    

    nearestCenter = None
    nearestDist = None
    
    for i in range(len(centers)):

        distance = 0

        for d in range(1, dim):

            dist = point[d] - centers[i][d]
            distance += dist**2

        distance = np.sqrt(distance)

        if nearestDist == None:
            nearestDist = distance
            nearestCenter = i

        elif nearestDist > distance:
            nearestDist = distance
            nearestCenter = i
            
    return nearestCenter
