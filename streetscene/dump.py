import sys
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
from rxpts_gen import *
# from rxpts_gen import samplelogx
# from rxpts_gen import dumptworayrunme
from envdumper import envdumper

def tx_main():
    l = []
    l += samplecube(nptr=20, xlow = 0, xhigh = 0, ylow = -15, yhigh = 15, zlow = 1, zhigh = 20)
    dump(l)
    return 0

def rx_main():
    l = samplelogx(nptr=50,xlow=80,xhigh=100000,ylow=-15,yhigh=15,zlow=1,zhigh=20)
    dump(l)
    return 0


def getEnvDump():
    # muBl = [20,100,250,500,1000]
    # muSl = [20,100,250,500,1000]
    # stdBl = [0.1, 0.2, 0.3]
    # stdSl = [0.1, 0.2, 0.3]
    muBl = [50,150,750,1250]
    muSl = [50,150,750,1250]
    stdBl = [0.15, 0.25]
    stdSl = [0.15, 0.25]
    widthl = [15]
    envfnl = []
    rxfnl = []
    txfnl = []
    envdir = "env/"
    rxdir = "rxpt/"
    txdir = "txpt/"
    for a in muBl:
        for b in muSl:
            for c in stdBl:
                for d in stdSl:
                    for e in widthl:
                        specstr = str(a) + "_"+ str(b) + "_"+ str(c) + "_"+ str(d) + "_" + str(e)
                        envfn = envdir + "env_streetscene_" + specstr
                        rxfn = rxdir + "rxpt_" + specstr
                        txfn = txdir + "txpt_" + specstr
                        dumpGaussianEnv(a,b,a*c,b*d,e, envfn, 80000)
                        envfnl.append(envfn)
                        rxfnl.append(rxfn)
                        txfnl.append(txfn)
    return [envfnl , rxfnl, txfnl]


def dumpGaussianEnv(muB, muS, stdB, stdS, width, envfn, xlimit = 10000):
    dumper = envdumper()
    dumper.xlimit = xlimit
    dumper.width = [-width,width]
    dumper.sampleGaussian(muB,muS,stdB,stdS)
    dumper.dump(envfn)


def dump_main(rxfn, txfn):
    envfnl, rxfnl, txfnl = getEnvDump()

    tout = sys.stdout
    for rxfn in rxfnl:
        f = open(rxfn, 'w')
        sys.stdout = f
        rx_main()
        f.close()
    for txfn in txfnl:
        f = open(txfn, 'w')
        sys.stdout = f
        tx_main()
        f.close()
    sys.stdout = tout
    print("DUMP: Rx file saved to " + rxfn)
    print("DUMP: Tx file saved to " + txfn)
    dumprunme(fnl=envfnl, rxfnl = rxfnl, txfnl = txfnl, rofp = "result_",
    fofp = "fields_",rofdir="rtresult/",fofdir="field/")
    print("DUMP: Env file saved to ./env/*")
    print("DUMP: runme dumped")
    return

if __name__ == '__main__':
    rxfn = 'streetscene_rx_pts.txt'
    txfn = 'streetscene_tx_pts.txt'
    # envfn = 'env_firstmodel.txt'
    dump_main(rxfn,txfn)
    #print(getEnvDump())
