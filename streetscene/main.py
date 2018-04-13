import sys
sys.path.append("/home/captainpenguins/thesis/thesis/algorithm")
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
import tensorflow as tf
import numpy as np
import os.path
import pickle
import matplotlib.pyplot as plt
from mydatautil import mydata
from mlp import mlp3L,mlp2L,mlp1L,mlp2Lhistplot


if __name__ == '__main__':
    # train models:

        # newpred = tworay('test',testmode = 0, epochs = 20000, breaklim = 15, plot = 1,
        # mlp_learning_rate = 0.05, traindata = 'tworaytrainset', testdata = 'tworaytestset',
        # plotrangel = 0, plotrangeh = 650, mlp_nbatchs = 3);

    pred = []
    testcost = []
    for i in range(0,3):
        histplot = True
        threelayers = False
        twolayers = False
        onelayer = False
        if histplot:
            newpred = mlp2Lhistplot('model'+str(i), testmode = 1, mlp_ninput = 14, epochs = 10000, breaklim = 34, plot = 1,
            mlp_learning_rate = 0.08, traindata = 'streetscene_trainset1', testdata = 'streetscene_testset2',
            plotrangel = 500, plotrangeh = 550, mlp_nbatchs = 4, mlp_nhidden1 = 100, mlp_nhidden2=50,
            plotTitle = '900MHz streetscene model');
        if threelayers:
            newpred = mlp3L('model3L'+str(i), testmode = 1, mlp_ninput = 14, epochs = 4000, breaklim = 32, plot = 0,
            mlp_learning_rate = 0.08, traindata = 'streetscene_trainset1', testdata = 'streetscene_testset1',
            plotrangel = 500, plotrangeh = 550, mlp_nbatchs = 4, mlp_nhidden1 = 100, mlp_nhidden2=50, mlp_nhidden3 = 30,
            plotTitle = '900MHz streetscene model');
        if twolayers:
            newpred = mlp2L('model'+str(i), testmode = 0, mlp_ninput = 14, epochs = 4000, breaklim = 34, plot = 1,
            mlp_learning_rate = 0.08, traindata = 'streetscene_trainset1', testdata = 'streetscene_testset1',
            plotrangel = 500, plotrangeh = 550, mlp_nbatchs = 4, mlp_nhidden1 = 100, mlp_nhidden2=50,
            plotTitle = '900MHz streetscene model');
        if onelayer:
            newpred = mlp1L('model1L'+str(i), testmode = 0, mlp_ninput = 14, epochs = 10000, breaklim = 35, plot = 0,
            mlp_learning_rate = 0.08, traindata = 'streetscene_trainset1', testdata = 'streetscene_testset3',
            plotrangel = 0, plotrangeh = 50, mlp_nbatchs = 4, mlp_nhidden1 = 100,
            plotTitle = '900MHz streetscene model');
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
