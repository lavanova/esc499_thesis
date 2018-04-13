import numpy as np

def readxls(rfn):
    # read the file
    m = 0
    n = -1
    with open(rfn,'r') as f:
        for line in f:
            m += 1
            newn = len(line.split(','))
            assert(newn == n or n == -1), "READXLS: number of colomns not consistent!"
            n = newn
    a = np.zeros((m,n))
    lc = 0
    with open(rfn,'r') as f:
        for line in f:
            nl = line.split(',')
            for j in range(n):
                a[lc][j] = float(nl[j])
            lc += 1
    return a

def readraytracerout(rfn, verbose = 0, indice = [2,3,4,5,6]):
    m = 0
    n = -1
    with open(rfn,'r') as f:
        for line in f:
            m += 1
            out = [s.strip() for s in line.split('\t') if s != '' and s != '\n']
            newn = len(out)
            assert(newn == n or n == -1), "READXLS: number of colomns not consistent!"
            n = newn
    assert(n!=-1),"READXLS: File is empty!"
    a = np.zeros((m,n))
    lc = 0
    with open(rfn,'r') as f:
        for line in f:
            nl = [s.strip() for s in line.split('\t') if s != '' and s != '\n']
            for j in range(n):
                a[lc][j] = float(nl[j])
            lc += 1
    if verbose:
        size = len(a)
        vinfo = rfn.split('_')
        for i in indice:
            a = np.append(a, np.full((size,1),float(vinfo[i])), axis = 1)

        # freq = np.full((size,1),float(vinfo[1]))
        # epsilon = np.full((size,1),float(vinfo[2]))
        # sigma = np.full((size,1),float(vinfo[3]))
        # a = np.append(a, freq, axis = 1)
        # a = np.append(a, epsilon, axis = 1)
        # a = np.append(a, sigma, axis = 1)
        #print(a[1])
    return a

if __name__ == '__main__':
    #a = readxls('firstmodel_rx_pts.txt')
    a = readraytracerout('/home/captainpenguins/thesis/tworay/tworayresult')
    print(a)
