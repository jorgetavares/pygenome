import numpy as np
import pygenome as pg


class Grammar(object):

    def __init__(self, filename=None):
        if filename is not None:
            self.grammar, self.start_symbol = self.parseFromFile(filename)
        else:
            self.grammar = {}
            self.start_symbol = None

    def parseFromFile(self, filename):

        def parse_line(line):
            rule = line.split(sep=':=')
            left_side = rule[0].strip()
            right_side = [x.strip() for x in rule[1].split(sep='|')]
            return left_side, right_side

        grammar = {}

        with open(filename, 'r') as input_file:
            left_side, right_side = parse_line(input_file.readline())
            grammar[right_side] = left_side
            start_symbol = left_side

            for line in input_file:
                left_side, right_side = parse_line(line)
                grammar[right_side] = left_side

        return grammar, start_symbol
