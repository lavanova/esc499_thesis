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
    for i in range(4):
        histplot = 0
        threelayers = False
        twolayers = 1
        onelayer = False
        if histplot:
            newpred = mlp2Lhistplot('model'+str(i),testmode = 1, mlp_ninput = 10, epochs = 10000, breaklim = 20, plot = 1,
            mlp_learning_rate = 0.06, traindata = 'uctrainsetr1', testdata = 'uctestsetr2',
            plotrangel = 150, plotrangeh = 200, mlp_nbatchs = 5,
            mlp_nhidden1 = 50, mlp_nhidden2=30, plotTitle = '900MHz urban canyon model', normalize = True);
        if threelayers:
            newpred = mlp3L('model3L'+str(i),testmode = 1, mlp_ninput = 10, epochs = 10000, breaklim = 18, plot = 0,
            mlp_learning_rate = 0.06, traindata = 'uctrainsetr1', testdata = 'uctestsetr3',
            plotrangel = 150, plotrangeh = 200, mlp_nbatchs = 5,
            mlp_nhidden1 = 50, mlp_nhidden2=30, mlp_nhidden3=20, plotTitle = '900MHz urban canyon model', normalize = True);
        if twolayers:
            newpred = mlp2L('model'+str(i),testmode = 1, mlp_ninput = 10, epochs = 10000, breaklim = 20, plot = 1,
            mlp_learning_rate = 0.06, traindata = 'uctrainsetr1', testdata = 'uctestsetr2',
            plotrangel = 1500, plotrangeh = 2000, mlp_nbatchs = 5,
            mlp_nhidden1 = 50, mlp_nhidden2=30, plotTitle = '900MHz urban canyon model', normalize = True);
        if onelayer:
            newpred = mlp1L('model1L'+str(i),testmode = 0, mlp_ninput = 10, epochs = 10000, breaklim = 26, plot = 0,
            mlp_learning_rate = 0.04, traindata = 'uctrainsetr1', testdata = 'uctestsetr3',
            plotrangel = 0, plotrangeh = 150, mlp_nbatchs = 4,
            mlp_nhidden1 = 50, plotTitle = '900MHz urban canyon model', normalize = True);

        pred.append(newpred[0])
        testcost.append(newpred[1])


        # except (RuntimeError):
        #     pass

    stdlist = []
    for j in range(len(pred[0])):
        stdtmp = []
        for i in range(len(pred)):
            stdtmp.append(pred[i][j])
        stdlist.append(np.std(stdtmp))
    print("The mean square error prediction between models is: ",np.mean(stdlist))
    print("The test cost for the models are: ",testcost)
