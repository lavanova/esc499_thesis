import sys
import random

class envdumper:
    def __init__(self):
        self.leftwallpos = []
        self.rightwallpos = []
        self.width = [-15,15]
        self.rel_perm = 5
        self.conductivity = 0.001
        self.xlimit = 10000
        self.L = -1
        self.I = -1
        self.muB = -1
        self.muS = -1
        self.stdB = -1
        self.stdS = -1

        self.filename = 'tmp'
        self.rofn = 'rofn'
        self.fofn = 'fofn'
        return

    def samplePoisson(self,L,I):
        self.L = L
        self.I = I
        # clear the original sample result
        self.leftwallpos = []
        self.rightwallpos = []
        pos = 0
        tmp = []
        for i in range (2):
            pos = 0
            if (i == 0):
                tmp = self.leftwallpos
            else:
                tmp = self.rightwallpos
            # decide whether it starts with a block or a space
            if (random.random() < I/(L+I)):
                pos = pos + random.expovariate(1/I)
            while (pos < self.xlimit):
                block = random.expovariate(1/L)
                tmp.append((pos,pos+block))
                pos += block
                space = random.expovariate(1/I)
                pos += space
        return


    def sampleGaussian(self, muB, muS, stdB, stdS):
        self.muB = muB
        self.muS = stdB
        self.stdB = muS
        self.stdS = stdS
        # clear the original sample result
        self.leftwallpos = []
        self.rightwallpos = []
        for i in range (2):
            pos = 0
            if (i == 0):
                tmp = self.leftwallpos
            else:
                tmp = self.rightwallpos
            # decide whether it starts with a block or a space
            if (random.random() < muS/(muB+muS)):
                pos = pos + random.gauss(muS,stdS)
            while (pos < self.xlimit):
                block = random.gauss(muB,stdB)
                tmp.append((pos,pos+block))
                pos += block
                space = random.gauss(muS,stdS)
                pos += space
        return

    def sampleTwoplates(self, muL, muLS, muR, muRS):
        # clear the original sample result
        self.leftwallpos = []
        self.rightwallpos = []
        for i in range (2):
            pos=0
            if (i == 0):
                tmp = self.leftwallpos
                pos = pos + random.expovariate(1/muLS)
                block = random.expovariate(1/muL)
            else:
                tmp = self.rightwallpos
                pos = pos + random.expovariate(1/muRS)
                block = random.expovariate(1/muR)
            tmp.append((pos,pos+block))
        return


    def dump(self,filename):
        self.filename = filename
        with open(filename, 'w') as wf:
            wf.write("recursion_level = 5;\ntransmitter_location = (0, 0, 0);\nreceiver_location = (10, 1, 1);\
            \nradiated_power = 1;\nsweep_param = dist;\nstart_frequency = 900;\nnumber_of_surfaces = 1;\nuse_pattern = 0;\
            \nenable_illum_zone = 0;\n// Ground\nsurfaceG0\n{\n  rel_perm = " + str(self.rel_perm) + ";\n  conductivity = " + str(self.conductivity) +";	// [S/m]\
            \nvertices = (0," + str(self.width[0]) + ",0) (0," + str(self.width[1]) + ",0) (100000," + str(self.width[1]) + ",0) (100000," + str(self.width[0]) + ",0);\n}\n")
            for i in range(len(self.leftwallpos)):
                xll = self.leftwallpos[i][0]
                xlh = self.leftwallpos[i][1]
                lwname = 'L' + str(i)
                wf.write("surface" + lwname + "\n{\n  rel_perm = " + str(self.rel_perm) + ";\n  conductivity = " + str(self.conductivity) +";\n")
                wf.write("vertices = ("+ str(xll) +"," + str(self.width[1]) + ",0) (" + str(xlh) + "," + str(self.width[1]) + ",0) (" + str(xlh) + "," + str(self.width[1]) + ",1000) (" + str(xll) + "," + str(self.width[1]) + ",1000);\n")
                wf.write("}\n")
            for j in range(len(self.rightwallpos)):
                xrl = self.rightwallpos[j][0]
                xrh = self.rightwallpos[j][1]
                rwname = 'R' + str(j)
                wf.write("surface" + rwname + "\n{\n  rel_perm = " + str(self.rel_perm) + ";\n  conductivity = " + str(self.conductivity) +";\n")
                wf.write("vertices = ("+ str(xrl) +"," + str(self.width[0]) + ",0) (" + str(xrh) + "," + str(self.width[0]) + ",0) (" + str(xrh) + "," + str(self.width[0]) + ",1000) (" + str(xrl) + "," + str(self.width[0]) + ",1000);\n")
                wf.write("}\n")
            self.rofn = 'results_' + filename
            self.fofn = 'fields_' + filename
            wf.write("\n // end automatic dump: " + filename)

if __name__ == "__main__":
    test1 = envdumper()
    test1.samplePoisson(100,50)
    test1.dump('test1')
