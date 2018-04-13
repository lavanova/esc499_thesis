import tensorflow as tf
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
from mydatautil import mydata
from mlp import mlp
import random
import math
from sklearn import linear_model

if __name__ == '__main__':
    # variables
    samplepts = 30
    xlow = 50
    xhigh = 100000
    dataname = 'getpathloss'
    h1 = 20
    h2 = 20

    # sample and save data
    h1 = math.log10(h1)
    h2 = math.log10(h2)
    xlowlog = math.log10(xlow)
    xhighlog = math.log10(xhigh)
    samplelist = []
    xlist = []
    ydummy = []
    for i in range(samplepts):
        # log value is sampled
        xslog = random.uniform(xlowlog, xhighlog)
        samplelist.append([h1,h2,xslog])
        xlist.append([xslog])
        ydummy.append([1])
    samplelist = np.asarray(samplelist)
    ydummy = np.asarray(ydummy)
    sampledata = mydata(samplelist,ydummy,dataname)
    sampledata.save()

    # generate model prediction
    result = mlp('model0',testmode = 1, epochs = 20000, breaklim = 15, plot = 0,
    mlp_learning_rate = 0.05, traindata = 'tworaytrainset', testdata = dataname,
    plotrangel = 0, plotrangeh = 650, mlp_nbatchs = 3,
    plotTitle = '900MHz two ray model');

    # linear fit
    #print(result[0])
    regr = linear_model.LinearRegression()
    regr.fit(xlist, result[0])
    yfit = regr.predict(xlist)
    plt.plot(xlist,result[0],'b^',xlist,yfit,'y')
    plt.show()
    print('Coefficients: \n', regr.coef_[0], regr.intercept_)
