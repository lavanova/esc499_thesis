import sys
import random
import math

# def tx_main():
#     l = []
#     l += sweep(0,0,1,'z',40,3)
#     #l += sweep(0,0,50,'z',150,10)
#     dump(l)
#     return 0
#
# def rx_main():
#     l = samplelogx(1000,100,10000,-15,15,1,40)
#     dump(l)
#     return 0

def twoplates_envdump():
    twoplatesdump(['test'], 15, [[5,100,7,75]])

def sweep(x,y,z,str,hlim,res):
    l = []
    if str == 'x':
        i = x
        while(i<hlim):
            l.append((i,y,z))
            i += res
    elif str == 'y':
        i = y
        while(i<hlim):
            l.append((x,i,z))
            i += res
    elif str == 'z':
        i = z
        while(i<hlim):
            l.append((x,y,i))
            i += res
    else:
        assert(0),"Invalid str input to sweep!"
    return l

def samplecube(nptr, xlow, xhigh, ylow, yhigh, zlow, zhigh):
    l = []
    for i in range(nptr):
        xs = random.uniform(xlow, xhigh)
        ys = random.uniform(ylow, yhigh)
        zs = random.uniform(zlow, zhigh)
        l.append((xs,ys,zs))
    return l

def samplelogx(nptr, xlow, xhigh, ylow, yhigh, zlow, zhigh):
    l = []
    xlowlog = math.log(xlow)
    xhighlog = math.log(xhigh)
    for i in range(nptr):
        xslog = random.uniform(xlowlog, xhighlog)
        ys = random.uniform(ylow, yhigh)
        zs = random.uniform(zlow, zhigh)
        xs = math.exp(xslog)
        l.append((xs,ys,zs))
    return l

def dump(l):
    for i in l:
        print(str(i[0]) + ',' + str(i[1]) + ',' + str(i[2]) + ',' + '0,0,1,0,0,0,1,0,0,0,1')
    return 0

def twoplatesdump(fnl, halfwidth, positions):
    '''
    (list(filenames), float, list of list(4 floats)) -> None
    dump two plates apart with a certain width. The plates have two degrees of freedom and here defined as xlow and xhigh
    '''
    assert(len(fnl) == len(positions)), "TWOPLATESDUMP: the size of name list and that of positions list does not match!";
    with open("runme",'w') as runme:
        for index in range (len(fnl)):
            fn = fnl[index];
            position = positions[index];
            xll = position[0];
            xlh = position[1];
            xrl = position[2];
            xrh = position[3];
            with open(fn, 'w') as wf:
                wf.write("recursion_level = 5;\ntransmitter_location = (0, 0, 0);\nreceiver_location = (10, 1, 1);\n radiated_power = 1;\nsweep_param = dist;\nstart_frequency = 900;\nnumber_of_surfaces = 1;\nuse_pattern = 0;\n// Ground\nsurfaceA\n{\n  rel_perm = 5;\n  conductivity = 0.001;	// [S/m]\n  vertices = (0,15,0) (0,-15,0) (10000,-15,0) (10000,15,0);\n}\n")
                wf.write("// Left Wall\nsurfaceB\n{\n  rel_perm = 5;\n  conductivity = 0.001;\n")
                wf.write("vertices = ("+ str(xll) +"," + str(halfwidth) + ",0) (" + str(xlh) + "," + str(halfwidth) + ",0) (" + str(xlh) + "," + str(halfwidth) + ",1000) (" + str(xll) + "," + str(halfwidth) + ",1000);\n")
                wf.write("}\n")
                wf.write("// Right Wall\nsurfaceC\n{\n  rel_perm = 5;\n  conductivity = 0.001;\n")
                wf.write("vertices = ("+ str(xrl) +"," + str(-halfwidth) + ",0) (" + str(xrh) + "," + str(-halfwidth) + ",0) (" + str(xrh) + "," + str(-halfwidth) + ",1000) (" + str(xrl) + "," + str(-halfwidth) + ",1000);\n")
                wf.write("}")
                rofn = 'results_' + fn
                fofn = 'fields_' + fn
            runme.write("../raytracer.out "+ fn + " -rec_loc=twoplates_rx_pts.txt -trans_loc=twoplates_tx_pts.txt -refs=4 -grho_h=0 > dump\n")
            runme.write("mv results.txt " + rofn + '\n')
            runme.write("mv fields.txt " + fofn + '\n')
    return

def dumptworayrunme(fn,rxfn,txfn,rofn='result.txt',fofn='fields.txt'):
    with open("runme",'w') as runme:
        runme.write("#!/bin/bash\n")
        runme.write("../raytracer.out "+ fn + " -rec_loc="+ rxfn +" -trans_loc="+ txfn +" -refs=2 -grho_h=0 > dump\n")
        runme.write("mv results.txt " + rofn + '\n')
        runme.write("mv fields.txt " + fofn + '\n')
    return
#def main():

def dumpucrunme(fn,rxfn,txfn,rofn='result.txt',fofn='fields.txt'):
    with open("runme",'w') as runme:
        runme.write("#!/bin/bash\n")
        runme.write("../raytracer.out "+ fn + " -rec_loc="+ rxfn +" -trans_loc="+ txfn +" -refs=2 -grho_h=0 > dump\n")
        runme.write("mv results.txt " + rofn + '\n')
        runme.write("mv fields.txt " + fofn + '\n')
    return

def dumprunme(fnl,rxfnl,txfnl,rofp = "result_",fofp = "fields_",envdir="env/",rofdir="result/",fofdir="field/"):
    with open("runme",'w') as runme:
        runme.write("#!/bin/bash\n")
        for i in range (len(fnl)):
            fn = fnl[i]
            rxfn = rxfnl[i]
            txfn = txfnl[i]
            runme.write("../raytracer.out " + fn + " -rec_loc="+ rxfn +" -trans_loc="+ txfn +" -refs=2 -grho_h=0 > dump/dump" + str(i) + "\n")
            runme.write("mv results.txt " + rofdir + rofp + fn.strip("/env_") + '\n')
            runme.write("mv fields.txt " + fofdir + fofp + fn.strip("/env_") + '\n')
    return

if __name__ == '__main__':
    f = open('twoplates_rx_pts.txt','w')
    sys.stdout = f
    rx_main()
    f.close()

    f = open('twoplates_tx_pts.txt','w')
    sys.stdout = f
    tx_main()
    f.close()

    #dumptworayrunme(fn='env_firstmodel.txt', rofn='tworayresult2', fofn='tworayfield2')
    #twoplates_envdump()
    #l = samplelogx(100,10,1000,-15,15,3,10)
    #print(l)
