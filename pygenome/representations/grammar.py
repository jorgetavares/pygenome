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
            rule = line.split(sep='::=')
            name = rule[0].strip()
            productions = [x.strip() for x in rule[1].split(sep='|')]
            return name, productions

        grammar = {}

        with open(filename, 'r') as input_file:
            name, productions = parse_line(input_file.readline())
            grammar[name] = productions
            start_symbol = name

            for line in input_file:
                name, productions = parse_line(line)
                grammar[name] = productions

        return grammar, start_symbol

    def randomDerivative(start_symbol):

        randomDrivative.program = ''

        def random_map(current_symbol):

            if current_symbol in self.grammar:
                productions = self.grammar[current_symbol]
                
                if type(productions) is list:
                    size = len(productions)
                    return random_map(productions[np.random.randint(size)])
                elif:
                    symbols = productions.split()
                    return " ".join([random_map(symbol) for symbol in productions])
            else:
                return current_symbol
