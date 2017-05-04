import numpy as np
import pygenome as pg

class Grammar(object):

    def __init__(self, filename=None):
        if filename is not None:
            self.grammar = self.parseFromFile(filename)
        else:
            self.grammar = {}

    def parseFromFile(self, filename):
        grammar = {}
        
        with open(filename, 'r') as input_file:
            for line in input_file:
                rule = line.split(sep=':=')
                right_side = rule[0].strip()
                left_side = [x.strip() for x in rule[1].split(sep='|')]
                grammar[right_side] = left_side
        
        return grammar
