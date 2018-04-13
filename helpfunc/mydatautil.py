import numpy as np
import os
import sys
import random
import pickle
sys.path.append("..")


class mydata:
    def __init__(self, x, y, model_name, scountlim =-1):
        self.x = x
        self.y = y
        self.size = len(x)
        self.insize = len(x[0])
        self.model_name = model_name
        self.isnormalized = False
        self.normalMean = np.mean(self.x,0)
        self.normalStd = np.std(self.x,0)
        assert(type(model_name) == type('')), "DATA INIT: model name not a string"
        assert((type(scountlim) == type(1) and scountlim > 0) or scountlim ==-1), "DATA INIT: scounter value fault"
        self._scountlim = scountlim
        self._cpt = 0
        self._scounter = 0
        random.seed(0)

        if len(x) != len(y):
            print("The input size: %d" % len(x))
            print("The label size: %d" % len(y))
            assert (len(x) == len(y)), "DATA INIT: number of inputs and labels does not match!"

    def isnormal(self):
        return self.isnormalized

    def normalize(self):
        assert(self.isnormalized == False), "DATA NORMALIZE: data is already normalized!"
        self.normalMean = np.mean(self.x,0)
        self.normalStd = np.std(self.x,0)
        self.x -= self.normalMean
        self.x /= self.normalStd
        self.isnormalized = True
        print("DATA NORMALIZE: data normalized, one sample of data is: ")
        print(self.x[1,:])
        return

    def denormalize(self):
        assert(self.isnormalized == True), "DATA NORMALIZE: data is already normalized!"
        self.x *= self.normalStd
        self.x += self.normalMean
        self.isnormalized = False
        print("DATA DENORMALIZE: data denormalized, one sample of data is: ")
        print(self.x[1,:])
        return

    def setnormalize(self,inx):
        assert(self.normalStd != []), "DATA SETNORMALIZED: normalStd is empty!"
        assert(np.size(inx,1) == np.size(self.x,1)), "DATA SETNORMALIZED: size problem!"
        #print(id(inx))
        inx -= self.normalMean
        #print(id(inx))
        inx /= self.normalStd
        #print(id(inx))
        print("DATA SETNORMALIZED: set the data to be normalized")
        return inx

    def setdenormalize(self,inx):
        assert(self.normalStd != []), "DATA SETDENORMALIZED: normalStd is empty!"
        assert(np.size(inx,1) == np.size(self.x,1)), "DATA SETDENORMALIZED: size problem!"
        #print(id(inx))
        inx *= self.normalStd
        #print(id(inx))
        inx += self.normalMean
        #print(id(inx))
        print("DATA DESETNORMALIZED: set the data to be denormalized")
        return inx

    def setscountlim(self, scountlim):
        self._scountlim = scountlim
        assert((type(scountlim) == type(1) and scountlim > 0) or scountlim ==-1), "DATA setscountlim: scounter value fault"
        self._scounter = 0
        return 0

    def shuffle(self):
        combined = list(zip(self.x, self.y))
        random.shuffle(combined)
        self.x, self.y = zip(*combined)
        return 0

    def nextbatch(self, batchsize):
        end = self._cpt + batchsize

        if end <= len(self.x):
            x = self.x[self._cpt:end]
            y = self.y[self._cpt:end]
        else:
            x_first = self.x[self._cpt:]
            y_first = self.y[self._cpt:]

            if self._scountlim != -1:
                self._scounter += 1
                self._scounter %= self._scountlim
                if self._scounter == 0:
                    self.shuffle()

            end = batchsize - (len(self.x) - self._cpt)
            x_second = self.x[:end]
            y_second = self.y[:end]

            x = []
            x.extend(x_first)
            x.extend(x_second)

            y = []
            y.extend(y_first)
            y.extend(y_second)

        assert(len(y) == batchsize) ,"DATA INITERNAL: size of output does not match batch size!"
        self._cpt = end
        return (np.array(x), np.array(y))

    def save(self, filename = ''):
        if filename == '':
            filename = self.model_name + '.pickle'
        save = pickle.dump(self, open(('./data/' + filename), 'wb'))
        return 0

    @staticmethod
    def load(filename):
        filename = './data/' + filename + '.pickle'
        print(filename)
        result = pickle.load(open(filename,'rb'))
        return result

    @staticmethod
    def loadfile(filename):
        result = pickle.load(open(filename,'rb'))
        return result

def std_data_process(data, indice, VAL, SMA, S):
    # performs:  (val - sma)/z normalization method
    assert(type(indice) == type([]) or type(indice) == type((1,))), "STD_DATA_PROCESS: indice not a list or tuple!"
    if len(indice) == 1:
        HIGH = indice[0]
        LOW = indice[0]
    elif len(indice) == 2:
        LOW = indice[0]
        HIGH = indice[1]
    else:
        assert(0), "STD_DATA_PROCESS: not supporting 3 or more arguments in indice"
    # DATA format has to be: [data_index][indicator][time]
    stddata = []
    for i in data:
        assert(type(i) == type([])), "STD_DATA_PROCESSS: Data is not a 2D or higher list!"
        nextdata = []
        for j in range(LOW,HIGH+1):
            nextdata.append((i[VAL][j] - i[SMA][j])/ 2*i[S][j])
        stddata.append(nextdata)
    return stddata


def discrete_sentiment_data_process(data, indice, VAL):
    # use -1/1 to represent the difference of the dimension of interest (VAL)
    assert(type(indice) == type([]) or type(indice) == type((1,))), "DISCRETE_SENTIMENT_DATA_PROCESS: indice not a list or tuple!"
    assert(len(indice) == 2), "DISCRETE_SENTIMENT_DATA_PROCESS: indice have to be length two! "
    assert(indice[1] >= indice[0] + 1), "DISCRETE_SENTIMENT_DATA_PROCESS: data has to be at least length 2! "
    LOW = indice[0]
    HIGH = indice[1]
    discrete_data = []
    for i in data:
        assert(type(i) == type([])), "DISCRETE_SENTIMENT_DATA_PROCESSS: Data is not a 2D or higher list!"
        nextdata = []
        for j in range(LOW,HIGH):
            if i[VAL][j+1] - i[VAL][j] >= 0:
                nextdata.append(1)
            else:
                nextdata.append(-1)
        discrete_data.append(nextdata)
    return discrete_data


def load_data_process(data, indice, VAL ,div = 1):
    assert(type(indice) == type([]) or type(indice) == type((1,))), "LOAD_DATA_PROCESS: indice not a list or tuple!"
    assert(len(indice) == 2), "LOAD_DATA_PROCESS: indice have to be length two! "
    assert(indice[1] >= indice[0]), "LOAD_DATA_PROCESS: second term should be greater or equal to the first term! "
    LOW = indice[0]
    HIGH = indice[1]
    load_data = []
    for i in data:
        assert(type(i) == type([])), "LOAD_DATA_PROCESS: Data is not a 2D or higher list!"
        nextdata = []
        for j in range(LOW,HIGH+1):
            nextdata.append(i[VAL][j]/div)
        load_data.append(nextdata)
    return load_data


def log_softmax_normalize(data, indice, VAL, softmax = False, mu = 0, stddev = 1):
    assert(type(indice) == type([]) or type(indice) == type((1,))), "LOG_SOFTMAX: indice not a list or tuple!"
    assert(len(indice) == 2), "LOG_SOFTMAX: indice have to be length two! "
    assert(indice[1] >= indice[0] + 1), "LOG_SOFTMAX: data has to be at least length 2! "
    LOW = indice[0]
    HIGH = indice[1]
    logdata = []
    for i in data:
        assert(type(i) == type([])), "DISCRETE_SENTIMENT_DATA_PROCESSS: Data is not a 2D or higher list!"
        nextdata = []
        for j in range(LOW,HIGH):
            logval = np.log(i[VAL][j+1] / i[VAL][j])
            if softmax:
                logval = 1 / (1 + np.exp( -(logval-mu)/stddev ))
            nextdata.append(logval)
        logdata.append(nextdata)
    return logdata


def data_checker(traindata,testdata):
    count = 0
    for i in range(len(testdata.x)):
        for j in range(len(traindata.x)):
            if listcmp(testdata.x[i], traindata.x[j]):
                count += 1
    return count


def listcmp(a,b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

if __name__ == '__main__':
    data = mydata.loadfile("/home/captainpenguins/thesis/thesis/tworay/data/tworaytrainset1.pickle")
    print(data.normalMean)
    data.normalize()
    data.denormalize()
    a = [[1,2,3]]
    a = data.setnormalize(a)
    print(a)
    a = data.setdenormalize(a)
    print(a)
    #day_cd_data_extract()
    #day_dd_data_extract()
