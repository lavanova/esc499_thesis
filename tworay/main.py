import sys
sys.path.append("/home/captainpenguins/thesis/thesis/algorithm")
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
import tensorflow as tf
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
from mydatautil import mydata
from mlp import mlp3L, mlp2L, mlp1L, mlp2Lhistplot


if __name__ == '__main__':
    # train models:

        # newpred = tworay('test',testmode = 0, epochs = 20000, breaklim = 15, plot = 1,
        # mlp_learning_rate = 0.05, traindata = 'tworaytrainset', testdata = 'tworaytestset',
        # plotrangel = 0, plotrangeh = 650, mlp_nbatchs = 3);

    pred = []
    testcost = []
    for i in range(0,3):
        histplot = False
        threelayers = False
        twolayers = True
        onelayer = False
        if histplot:
            newpred = mlp2Lhistplot('model'+str(i),testmode = 1, mlp_ninput = 3, epochs = 10000, breaklim = 1.8, plot = 1,
            mlp_learning_rate = 0.05, traindata = 'tworaytrainset1', testdata = 'tworaytestset2',
            plotrangel = 40, plotrangeh = 90, mlp_nbatchs = 3,
            mlp_nhidden1 = 30, mlp_nhidden2=20, plotTitle = '900MHz two ray model', normalize = True);
        if threelayers:
            newpred = mlp3L('model3L'+str(i),testmode = 1, mlp_ninput = 3, epochs = 10000, breaklim = 1.3, plot = 0,
            mlp_learning_rate = 0.05, traindata = 'tworaytrainset1', testdata = 'tworaytestset1',
            plotrangel = 40, plotrangeh = 90, mlp_nbatchs = 3,
            mlp_nhidden1 = 30, mlp_nhidden2=20, mlp_nhidden3=10, plotTitle = '900MHz two ray model', normalize = True);
        if twolayers:
            newpred = mlp2L('model'+str(i),testmode = 0, mlp_ninput = 3, epochs = 10000, breaklim = 3, plot = 0,
            mlp_learning_rate = 0.05, traindata = 'tworaytrainseth301', testdata = 'tworaytestseth303',
            plotrangel = 40, plotrangeh = 90, mlp_nbatchs = 3,
            mlp_nhidden1 = 30, mlp_nhidden2=20, plotTitle = '900MHz two ray model', normalize = True);
        if onelayer:
            newpred = mlp1L('model1L'+str(i),testmode = 1, mlp_ninput = 3, epochs = 10000, breaklim = 4, plot = 1,
            mlp_learning_rate = 0.07, traindata = 'tworaytrainset1', testdata = 'tworaytestset3',
            plotrangel = 40, plotrangeh = 90, mlp_nbatchs = 3,
            mlp_nhidden1 = 50, plotTitle = '900MHz two ray model', normalize = True);
        pred.append(newpred[0])
        testcost.append(newpred[1])


    stdlist = []
    for j in range(len(pred[0])):
        stdtmp = []
        for i in range(len(pred)):
            stdtmp.append(pred[i][j])
        stdlist.append(np.std(stdtmp))
    print("The mean square error prediction between models is: ",np.mean(stdlist))
    print("The test cost for the models are: ",testcost)
