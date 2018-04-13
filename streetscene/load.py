import sys
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
import numpy as np
import pickle
import random
from mydatautil import mydata
from readxls import readraytracerout
import os

def load_main(rfl, resultdir = "rtresult/", testper = 0.08, verbose = 0, traindataname = 'traindata', testdataname = 'testdata'):
    for i in range(len(rfl)):
        rfl[i] = resultdir + rfl[i]
    return load(rfl, testper, verbose, traindataname, testdataname)

def normalize(vlist):
    means = []
    stds = []
    for i in range (len(vlist)):
        print(vlist[i][1])
        mean = np.mean(vlist[i])
        std = np.std(vlist[i])
        vlist[i] -= mean
        vlist[i] /= std
        means.append(mean)
        stds.append(std)
        print(vlist[i][1])
    return [means,stds]

def load(rfl, testper = 0.08, verbose = 0, traindataname = 'traindata', testdataname = 'testdata'):
    a = []
    for rfn in rfl:
        b = readraytracerout(rfn, verbose = verbose, indice = [2,3,4,5,6])
        if a == []:
            a = b
        else:
            a = np.append(a,b,axis=0)

    # inputs
    ty = a[:,1]
    ty1 = ty + 15
    ty2 = 15 - ty
    tz = a[:,2]
    rx = a[:,3]
    ry = a[:,4]
    ry1 = ry + 15
    ry2 = 15 - ry
    rz = a[:,5]
    # +2 relationship
    y = a[:,10] # result in dB
    muB = a[:,12]
    muS = a[:,13]
    stdB = a[:,14]
    stdS = a[:,15]
    width = a[:,16]

    dg = np.sqrt( (ty - ry) ** 2 + (tz-rz) ** 2 + rx ** 2)
    dl = np.sqrt( (ty1 - ry1) ** 2 + (tz-rz) ** 2 + rx ** 2 )
    dr = np.sqrt( (ty2 - ry2) ** 2 + (tz-rz) ** 2 + rx ** 2 )
    lty1, lty2, lry1, lry2 = np.log10([ty1,ty2,ry1,ry2])
    ltz, lrz, lrx, ldg, ldl, ldr = np.log10([tz,rz,rx,dg,dl,dr])
    muB = np.log10(muB)
    muS = np.log10(muS)
    width = np.log10(width)

    #[lty1,lty2,lrx,ltz,lry1,lry2,lrz,ldg,ldl,ldr, muB,muS,stdB,stdS]

    #normalize([lty1,lty2,lrx,ltz,lry1,lry2,lrz,ldg,ldl,ldr, muB,muS,stdB,stdS])
    x = np.stack((lty1,lty2,lrx,ltz,lry1,lry2,lrz,ldg,ldl,ldr, muB,muS,stdB,stdS))

    x = x.T
    y = -y

    xtrain, ytrain, xtest, ytest = seperate_test_set(x,y,testper)
    traindata = mydata(xtrain,ytrain,traindataname)
    testdata = mydata(xtest,ytest,testdataname)
    print("LOAD: Training data size is: ", ytrain.size)
    print("LOAD: Testing data size is: ", ytest.size)
    print("LOAD: One sample of the data is:\n")
    print(str(xtrain[10]) + '      '   + str(ytrain[10]))

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
    rfl = os.listdir("./rtresult")
    print(rfl)
    load_main(rfl, resultdir = "rtresult/", testper = 0.8, verbose = 1,
     traindataname = 'streetscene_trainset3', testdataname = 'streetscene_testset3')
    #load(['tworayresult'], testper = 0.1, verbose = 0, traindataname = 'tworaytrainset', testdataname = 'tworaytestset')
