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


    def randomDerivative(self, current_symbol, isproductions=False):
        
        if current_symbol in self.grammar:
            productions = self.grammar[current_symbol]

            if type(productions) is list:
                size = len(productions)
                return self.randomDerivative(productions[np.random.randint(size)], isproductions=True)
            else:
                symbols = productions.split()
                return " ".join([self.randomDerivative(symbol) for symbol in symbols], isproductions=True)
        elif isproductions is True:
            symbols = current_symbol.split()
            return " ".join([self.randomDerivative(symbol) for symbol in symbols])
        else:
            return current_symbol


    def mapDerivative(self, start_symbol, values, wrap=True):

        def run_map(current_symbol, isproductions=False):
            if current_symbol in self.grammar:
                productions = self.grammar[current_symbol]

                if type(productions) is list:
                    size = len(productions)
                    run_map.position += 1
                    if run_map.position == values.size and wrap is True:
                        run_map.position = 0
                    return run_map(productions[np.mod(values[run_map.position], size)], isproductions=True)
                else:
                    symbols = productions.split()
                    return " ".join([run_map(symbol) for symbol in symbols], isproductions=True)
            elif isproductions is True:
                symbols = current_symbol.split()
                return " ".join([run_map(symbol) for symbol in symbols])
            else:
                return current_symbol
        
        run_map.position = -1
        program = run_map(start_symbol)
        return program
