import sys
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
import numpy as np
import pickle
import random
from mydatautil import mydata
from readxls import readraytracerout

def load_main(rfl, testper = 0.08, verbose = 0, traindataname = 'traindata', testdataname = 'testdata',extend = 0):
    if extend == 0:
        return load(rfl, testper, verbose, traindataname, testdataname)
    else:
        return load5in(rfl, testper, verbose, traindataname, testdataname)


def load(rfl, testper = 0.08, verbose = 0, traindataname = 'traindata', testdataname = 'testdata'):
    a = []
    for rfn in rfl:
        a = readraytracerout(rfn, verbose = verbose)
    h1 = a[:,2] # height of the source
    d = a[:,3] # distance
    h2 = a[:,5] # height of the receiver
    y = a[:,8] # result in dB

    if verbose:
        freq = a[:,-3]
        epsilon = a[:,-2]
        sigma = a[:,-1]

    d = np.sqrt(d*d+(h1-h2)*(h1-h2))
    d = np.log10(d)
    h1 = np.log10(h1)
    h2 = np.log10(h2)

    dstd = np.std(d)
    h1std = np.std(h1)
    h2std = np.std(h2)
    print("Std. of d is: ",dstd)
    print("Std. of h1 is: ",h1std)
    print("Std. of h2 is: ",h2std)
    # correct variance:
    h1cor = dstd/h1std
    h2cor = dstd/h2std
    #h1 = h1*h1cor
    #h2 = h2*h2cor

    exp1 = 40*d-20*h1-20*h2

    x = np.stack((h1,h2,d))
    #x = np.stack((h1,h2,d))
    x = x.T
    y = -y

    xtrain, ytrain, xtest, ytest = seperate_test_set(x,y,testper)
    traindata = mydata(xtrain,ytrain,traindataname)
    testdata = mydata(xtest,ytest,testdataname)
    print("LOAD: Training data size is: ", ytrain.size)
    print("LOAD: Testing data size is: ", ytest.size)
    print("LOAD: One sample of the data is:\n")
    print(str(xtrain[1:10]) + '      '   + str(ytrain[1]))

    traindata.save()
    testdata.save()
    print("LOAD: Saved traindata to ./data/" + traindataname)
    print("LOAD: Saved testdata to ./data/" + testdataname)
    return 0

def load5in(rfl, testper = 0.08, verbose = 0, traindataname = 'traindata', testdataname = 'testdata'):
    a = []
    for rfn in rfl:
        a = readraytracerout(rfn, verbose = verbose)
    h1r = a[:,2] # height of the source
    d = a[:,3] # distance
    h2r = a[:,5] # height of the receiver
    y = a[:,8] # result in dB

    #d = np.sqrt(d*d+(h1-h2)*(h1-h2))
    d = np.log10(d)
    h1 = np.log10(h1r)
    h2 = np.log10(h2r)

    #normalize everything
    dmean = np.mean(d)
    h1mean = np.mean(h1)
    h2mean = np.mean(h2)
    h1rmean = np.mean(h1r)
    h2rmean = np.mean(h2r)

    dstd = np.std(d)
    h1std = np.std(h1)
    h2std = np.std(h2)
    h1rstd = np.std(h1r)
    h2rstd = np.std(h2r)

    # correct variance:
    # d = (d - dmean) / dstd
    # h1 = (h1 - h1mean) / h1std
    # h2 = (h2 - h2mean) / h2std
    # h1r = (h1r - h1rmean) / h1rstd
    # h2r = (h2r - h2rmean) / h2rstd

    #x = np.stack((h1,h2,d))
    x = np.stack((h1,h2,d,h1r,h2r))
    exp1 = 40*d-20*h1-20*h2
    x = x.T
    y = -y

    xtrain, ytrain, xtest, ytest = seperate_test_set(x,y,testper)
    traindata = mydata(xtrain,ytrain,traindataname)
    testdata = mydata(xtest,ytest,testdataname)
    print("LOAD: Training data size is: ", ytrain.size)
    print("LOAD: Testing data size is: ", ytest.size)
    print("LOAD: One sample of the data is:\n")
    print(str(xtrain[1]) + '      '   + str(ytrain[1]))

    traindata.save()
    testdata.save()
    print("LOAD: Saved traindata to ./data/" + traindataname)
    print("LOAD: Saved testdata to ./data/" + testdataname)
    return 0

def seperate_test_set(x,y,testper,ignore = 0):
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    for i in range(len(x)):
        skip = random.random() < ignore
        if ignore:
            continue
        istest = random.random() < testper
        if istest:
            xtest.append(x[i])
            ytest.append([y[i]])
        else:
            xtrain.append(x[i])
            ytrain.append([y[i]])
    print("LOAD: Seperated " + str(testper*100) + " percent data forms the test set")
    return np.array(xtrain), np.array(ytrain), np.array(xtest), np.array(ytest)

if __name__ == '__main__':
    load(['tworayresult'], testper = 0.1, verbose = 0, traindataname = 'tworaytrainset1', testdataname = 'tworaytestset1')
    #load(['tworayresult'], testper = 0.1, verbose = 0, traindataname = 'tworaytrainset', testdataname = 'tworaytestset')
