import sys
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
from rxpts_gen import *
# from rxpts_gen import samplelogx
# from rxpts_gen import dumptworayrunme
from envdumper import envdumper

def tx_main():
    l = []
    l += samplecube(nptr=40, xlow = 0, xhigh = 0, ylow = 0, yhigh = 0, zlow = 1, zhigh = 30)
    dump(l)
    return 0

def rx_main():
    l = samplelogx(nptr=150,xlow=100,xhigh=10000,ylow=0,yhigh=0,zlow=1,zhigh=30)
    dump(l)
    return 0

def dump_main(rxfn, txfn, envfn):
    tout = sys.stdout
    f = open(rxfn, 'w')
    sys.stdout = f
    rx_main()
    f.close()

    f = open(txfn, 'w')
    sys.stdout = f
    tx_main()
    f.close()

    sys.stdout = tout

    print("DUMP: Rx file saved to " + rxfn)
    print("DUMP: Tx file saved to " + txfn)
    dumptworayrunme(fn=envfn, rxfn = rxfn, txfn = txfn, rofn='tworayresult', fofn='tworayfield')
    print("DUMP: Env file saved to " + envfn)
    print("DUMP: runme dumped")
    return

if __name__ == '__main__':
    rxfn = 'tworay_rx_pts.txt'
    txfn = 'tworay_tx_pts.txt'
    envfn = 'env_firstmodel.txt'
    dump_main(rxfn,txfn,envfn)
