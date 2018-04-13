import sys
sys.path.append("/home/captainpenguins/thesis/thesis/helpfunc")
from rxpts_gen import *
# from rxpts_gen import samplelogx
# from rxpts_gen import dumptworayrunme
from envdumper import envdumper

def tx_main():
    l = []
    #l += sweep(x=0,y=0,z=1,str='z',hlim=40,res=3)
    l += samplecube(nptr = 100, xlow = 0, xhigh = 0, ylow = -15, yhigh = 15, zlow = 1, zhigh = 20)
    dump(l)
    return 0

def rx_main():
    l = samplelogx(nptr=500,xlow=80,xhigh=100000,ylow=-15,yhigh=15,zlow=1,zhigh=20)
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
    dumpucrunme(fn=envfn, rxfn = rxfn, txfn = txfn, rofn='urbancanyonresult', fofn='urbancanyonfield')
    print("DUMP: Env file saved to " + envfn)
    print("DUMP: runme dumped")
    return

if __name__ == '__main__':
    rxfn = 'urbancanyon_rx_pts.txt'
    txfn = 'urbancanyon_tx_pts.txt'
    envfn = 'env_urbancanyon.txt'
    dump_main(rxfn,txfn,envfn)
