import pandas
from sklearn.decomposition import PCA

def normalize(data):
    """
    Function that gets a data column as a list and returns it with normalized values
    using the minmax normalization.
    """
    maxVal = max(data)
    minVal = min(data)
    data = (data - minVal) / (maxVal - minVal)
    return data

def weighted_average(data, indexes, weights):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and a list of weights for
    each column and returns the dataframe with the WA.
    """
    df = data
    value = 0

    for i in range(0, len(indexes)):
        value = value + (weights[i] * df[indexes[i]]) / sum(weights)

    df['WA'] = value
    return df.sort_values(['WA'], ascending = False)

def maximin(data, indexes):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and returns the dataframe
    with the maximin applied.
    """
    df = data
    values = []
    minVal = 0
    for i in range(0, len(df[indexes[0]])):
        minVal = df[indexes[0]][i]
        for j in indexes:
            if minVal > df[j][i]:
                minVal = df[j][i]
        
        values.append(minVal)
    df['minVal'] = pandas.DataFrame(values)

    return df.sort_values(['minVal'], ascending = False)

def minimax(data, indexes):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and returns the dataframe
    with the minimax applied.
    """
    df = data
    values = []
    maxVal = 0
    for i in range(0, len(df[indexes[0]])):
        maxVal = df[indexes[0]][i]
        for j in indexes:
            if maxVal < df[j][i]:
                maxVal = df[j][i]
        
        values.append(maxVal)
    df['maxVal'] = pandas.DataFrame(values)

    return df.sort_values(['maxVal'], ascending = True)

def leximin(data, indexes):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and returns the dataframe
    with the leximin applied.
    """
    df = data
    values = []
    lex = []
    columns = []

    for i in range(0, len(df[indexes[0]])):
        values.clear()
        for j in indexes:
            values.append(df[j][i])
        
        lex.append(sorted(values,reverse=False))

    for i in range(0, len(indexes)):

        values.clear()
        for j in range(0, len(df[indexes[0]])):
            values.append(lex[j][i])

        st = "C" + str(i+1)
        columns.append(st)
        df[st] = pandas.DataFrame(values)
    
    return df.sort_values(columns, ascending=False)

def leximax(data, indexes):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and returns the dataframe
    with the leximax applied.
    """
    df = data
    values = []
    lex = []
    columns = []

    for i in range(0, len(df[indexes[0]])):
        values.clear()
        for j in indexes:
            values.append(df[j][i])
        
        lex.append(sorted(values,reverse=True))

    for i in range(0, len(indexes)):

        values.clear()
        for j in range(0, len(df[indexes[0]])):
            values.append(lex[j][i])

        st = "C" + str(i+1)
        columns.append(st)
        df[st] = pandas.DataFrame(values)
    
    return df.sort_values(columns, ascending=True)

def ParetoDomina(a,b):
    mi = len([1 for i in range(len(a)) if a[i] >= b[i]])
    my = len([1 for i in range(len(a)) if a[i] > b[i]])
    if mi == len(a):
        if my > 0:
            return True
    return False

def skylines(data, indexes):
    """
    Function that gets a dataframe, a list of column names as indexes as strings and returns the dataframe
    with the leximax applied.
    """
    df = data
    t = df.shape[0]
    for i in range(t):
        if i in df.index:
            a = [0] * len(indexes)
            for j in range(i + 1, t):
                if j in df.index:
                    b = [0] * len(indexes)
                    for k in range(len(indexes)):
                        a[k] = df[indexes[k]][i]
                        b[k] = df[indexes[k]][j]
                    if ParetoDomina(a,b):
                        df = df.drop(j)
                    elif ParetoDomina(b,a):
                        df = df.drop(i)
                        break
    return df

def pca(data, n_comp):
    """
    Function that gets a dataframe and the number of components you want to reduce the dimension of your dataframe to 
    and returns a list with the n dimension.
    """
    pca = PCA(n_components=n_comp)
    pca.fit(data)

    x_pca = pca.transform(data)

    return x_pca