import os
import sys
import numpy as np

sys.path.append(os.path.abspath('..'))
import GeneticAlgorithm as GA

def test_crossover():
    genome1 = [0,1,2,3,4,5,6,7,8,9]
    genome2 = [10,11,12,13,14,15,16,17,18,19]

    result1 = [0,1,2,3,4,15,16,17,18,19]
    result2 = [10,11,12,13,14,5,6,7,8,9]

    assert((result1 == GA.crossover(genome1,genome2)).all())
    assert((result2 == GA.crossover(genome2,genome1)).all())

def test_randomMixing():
    genomeSet = [[0,1,2,3,4,5,6,7,8,9],[10,11,12,13,14,15,16,17,18,19]]
    scores    = [10,0]

    print((genomeSet[0] != GA.mixRandomly(genomeSet,scores,cmp=lambda a,b: a < b)[0]))
    print((genomeSet[1] == GA.mixRandomly(genomeSet,scores,cmp=lambda a,b: a < b)[1]))
    assert(not (genomeSet[0] == GA.mixRandomly(genomeSet,scores,cmp=lambda a,b: a < b)[0]).all())
    assert((genomeSet[1] == GA.mixRandomly(genomeSet,scores,cmp=lambda a,b: a < b)[1]).all())
