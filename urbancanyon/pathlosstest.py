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
import random
import math
from sklearn import linear_model
from execute import execute_main
from rxpts_gen import *
from readxls import readraytracerout

def tx_main(ty,tz):
    l = []
    #l += sweep(x=0,y=0,z=1,str='z',hlim=40,res=3)
    l += samplecube(nptr = 1, xlow = 0, xhigh = 0, ylow = ty, yhigh = ty, zlow = tz, zhigh = tz)
    dump(l)
    return 0

def rx_main(ry,rz):
    l = samplelogx(nptr=300,xlow=100,xhigh=10000,ylow=ry,yhigh=ry,zlow=rz,zhigh=rz)
    dump(l)
    l = np.asarray(l)
    return np.log10(l[:,0])

def dump_main(rxfn, txfn, envfn, ty, tz, ry, rz):
    tout = sys.stdout
    f = open(rxfn, 'w')
    sys.stdout = f
    xray = rx_main(ry, rz)
    f.close()
    f = open(txfn, 'w')
    sys.stdout = f
    tx_main(ty, tz)
    f.close()
    sys.stdout = tout
    print("DUMP: Rx file saved to " + rxfn)
    print("DUMP: Tx file saved to " + txfn)
    dumpucrunme(fn=envfn, rxfn = rxfn, txfn = txfn, rofn='plresult', fofn='plfield')
    print("DUMP: runme dumped")
    return xray

if __name__ == '__main__':
    # variables
    # lty1,lty2,lrx,ltz,lry1,lry2,lrz,ldg,ldl,ldr
    samplepts = 200
    xlow = 100
    xhigh = 10000
    dataname = 'getpathloss'
    ty = -10
    tz = 5
    ry = 10
    rz = 5

    ###
    ty1 = ty + 15
    ty2 = 15 - ty
    ry1 = ry + 15
    ry2 = 15 - ry

    lty1, lty2, lry1, lry2 = np.log10([ty1,ty2,ry1,ry2])

    # sample and save data
    xlowlog = math.log10(xlow)
    xhighlog = math.log10(xhigh)
    samplelist = []
    xlist = []
    ydummy = []
    for i in range(samplepts):
        xslog = random.uniform(xlowlog, xhighlog)
        dg = np.sqrt( (ty - ry) ** 2 + (tz-rz) ** 2 + (10 ** xslog) ** 2)
        dl = np.sqrt( (ty1 - ry1) ** 2 + (tz-rz) ** 2 + (10 ** xslog) ** 2 )
        dr = np.sqrt( (ty2 - ry2) ** 2 + (tz-rz) ** 2 + (10 ** xslog) ** 2 )
        ltz, lrz, ldg, ldl, ldr = np.log10([tz,rz,dg,dl,dr])
        # log value is sampled
        samplelist.append([lty1,lty2,xslog,ltz,lry1,lry2,lrz,ldg,ldl,ldr])
        #samplelist.append([h1,h2,xslog])
        xlist.append([xslog])
        ydummy.append([1])
    print(samplelist[0:2])
    samplelist = np.asarray(samplelist)
    ydummy = np.asarray(ydummy)
    sampledata = mydata(samplelist,ydummy,dataname)
    sampledata.save()

    # generate model prediction
    # result = mlp2L('model0',testmode = 1, mlp_ninput = 10, epochs = 10000, breaklim = 20, plot = 0,
    # mlp_learning_rate = 0.06, traindata = 'uctrainsetr1', testdata = dataname,
    # plotrangel = 150, plotrangeh = 200, mlp_nbatchs = 5,
    # mlp_nhidden1 = 50, mlp_nhidden2=30, plotTitle = '900MHz urban canyon model', normalize = True);

    result = mlp3L('model3L0',testmode = 1, mlp_ninput = 10, epochs = 10000, breaklim = 18, plot = 0,
    mlp_learning_rate = 0.06, traindata = 'uctrainsetr1', testdata = dataname,
    plotrangel = 150, plotrangeh = 200, mlp_nbatchs = 5,
    mlp_nhidden1 = 50, mlp_nhidden2=30, mlp_nhidden3=20, plotTitle = '900MHz urban canyon model', normalize = True);
    # linear fit
    #print(result[0])
    combined = list(zip(xlist, result[0]))
    combined.sort()
    xlist, result[0] = zip(*combined)

    # get raytracer result
    xray = dump_main(rxfn = 'ucpl_rx_pts.txt', txfn = 'ucpl_tx_pts.txt',
    envfn = 'env_urbancanyon.txt', ty = ty, tz = tz, ry = ry, rz = rz)
    execute_main()
    a = readraytracerout('plresult', verbose=0)
    yray = -a[:,11]

    combined = list(zip(xray, yray))
    combined.sort()
    xray, yray = zip(*combined)


    regr = linear_model.LinearRegression()
    regr.fit(xlist, result[0])
    yfit = regr.predict(xlist)
    plt.suptitle('Urban canyon path loss curve')
    plt.xlabel('log10(x)')
    plt.ylabel('negative signal strength in db')
    plt.plot(xlist,result[0],'b',xlist,yfit,'y',xray,yray,'r')
    plt.legend(['Predicted','Linear fit','Raytracer'])
    plt.show()
    print('Coefficients: \n', regr.coef_[0], regr.intercept_)
