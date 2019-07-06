data = []
labels = []
species = []

def getColumn(data, i):
    return [row[i] for row in data]

def replaceColumn(data, newData, i):
    for pos in range(len(data)):
        data[pos][i] = newData[pos]

    return data

def normalize(data):
    """
    Data normalization function that gets a dataset readed as a list of lists
    and returns the same dataset with the normalized data.
    """
    minVal = min(data)
    maxVal = max(data)
    result = list(map(lambda x: (x - minVal) / (maxVal - minVal), data))
    return result

def WA(data,weights):
    """
    Function that gets a dataset readed as a list of lists and a list of weights for
    each column and returns the dataframe with the WA.
    """
    total = sum(weights)
    newData = []
    for row in data:
        val = 0
        for i in range(len(row)):
            val = val + ((row[i] * weights[i])/total)
        row.append(val)
        newData.append(row)
    return newData

def maximin(data):
    """
    Function that gets a dataset readed as a list of lists and returns the dataset
    with the maximin applied
    """
    newData = []
    pos = len(data[0])
    for row in data:
        minVal = min(row)
        row.append(minVal)
        newData.append(row)

    def takeKey(elem):
        return elem[pos]

    newData.sort(key=takeKey, reverse = True)
    return newData

def minimax(data):
    """
    Function that gets a dataset readed as a list of lists and returns the dataset
    with the minimax applied
    """
    newData = []
    pos = len(data[0])
    for row in data:
        maxVal = max(row)
        row.append(maxVal)
        newData.append(row)

    def takeKey(elem):
        return elem[pos]

    newData.sort(key=takeKey, reverse = False)
    return newData

def leximin(data):
    """
    Function that gets a dataset readed as a list of lists and returns the dataset
    with the leximin applied
    """

    newData = []

    for row in data:
        arr = row.copy()
        arr.sort()
        newData.append(row + arr)

    def takeKey(elem):
        total = int(len(elem)/2)
        key = []

        for i in range(total,len(elem)):
            key.append(elem[i])
        return key

    newData.sort(key=takeKey, reverse = True)

    return newData

def leximax(data):
    """
    Function that gets a dataset readed as a list of lists and returns the dataset
    with the leximax applied
    """

    newData = []

    for row in data:
        arr = row.copy()
        arr.sort(reverse = True)
        newData.append(row + arr)

    def takeKey(elem):
        total = int(len(elem)/2)
        key = []

        for i in range(total,len(elem)):
            key.append(elem[i])
        return key

    newData.sort(key=takeKey, reverse = False)

    return newData

    with open('iris.csv') as csvfile:
    reader = csv.reader(csvfile)
    cols = [1,2,3,4]
    for row in reader:

        if(not labels):
            newRow = list(row[i] for i in cols)
            labels = newRow
        else:
            newRow = list(float(row[i]) for i in cols)
            data.append(newRow)
            species.append(row[5])

norm = normalize(getColumn(data,0))
replaceColumn(data,norm,0)

norm = normalize(getColumn(data,1))
replaceColumn(data,norm,1)

norm = normalize(getColumn(data,2))
replaceColumn(data,norm,2)

norm = normalize(getColumn(data,3))
replaceColumn(data,norm,3)
