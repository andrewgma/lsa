# modified from: http://www.puffinwarellc.com/lsa.py

from numpy import zeros
from scipy.linalg import svd
#following needed for TFIDF
from math import log
from numpy import asarray, sum

import sys
import csv

moods = sys.argv[1]

class LSA(object):

    def __init__(self):
        self.wdict = {}
        self.dcount = 0  
        self.wordlist = []

    def parse(self, doc):
        words = doc.split();
        for w in words:
            if w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
                self.wordlist.append(w)
        self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i,d] += 1

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)

    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)        
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])
   
    def makeCSV(self):
    	c=csv.writer(open("moods.csv","wb"))
    	for i in range(len(self.wordlist)):
    		c.writerow([self.wordlist[i],-1*self.U[i][0],-1*self.U[i][1],-1*self.U[i][2],-1*self.U[i][4]])


mylsa = LSA()
for line in open(moods,'r'):
    mylsa.parse(line)
mylsa.build()
mylsa.TFIDF()
mylsa.calc()
mylsa.makeCSV()