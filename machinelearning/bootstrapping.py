import pandas as pd
import numpy as np

def getIndMatrix(barIx, t1):
    indM = pd.DataFrame(0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1,i] = i
    
    return indM

def getAvgUniqueness(indM):
    c = indM.sum(axis=1) # Concurrency
    u = indM.div(c, axis=0) #Uniqueness
    avgU = u[u > 0].mean() # Average Uniqueness

    return avgU

def seqBootstrap(indM, sLength=None):
    # Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:  
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi+[i]] # reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]

        prob = avgU/avgU.sum() # Draw Prob
        phi += [np.random.choice(indM.columns, p=prob)]
    return phi
