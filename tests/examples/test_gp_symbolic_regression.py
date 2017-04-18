import sys
sys.path.append('../../')

from  examples.gp_symbolic_regression import *

stdout = """0	46.5	156.465962269	183.617385551
1	46.5	129.515668368	26.4457168862
2	32.5	118.940807937	26.359443112
3	32.5	109.209110317	24.2065905232
4	32.5	101.450630556	28.8134975899
5	32.5	95.8124420885	33.0380205987
6	32.5	90.8682071429	36.4060780524
7	0.0	87.0197492063	42.0129327554
8	0.0	84.6285174603	64.4940287076
9	0.0	82.6545202381	88.168072054
10	0.0	76.3909158388	72.2735948096
11	0.0	68.0289865079	56.9330786656
12	0.0	67.1032607143	75.5617114031
13	0.0	59.773565873	70.3779848367
14	0.0	56.0148462302	53.5544621083
15	0.0	55.1444109668	58.4136290623
16	0.0	53.9641391414	62.6112642487
17	0.0	52.7631607143	61.4085895057
18	0.0	57.1935546232	70.2937498985
19	0.0	55.5348072025	74.3467864017
fitness: 0.0	genome: [ 3  4 11  1  6  9 11  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
"""

def test_gp_with_elitism(capfd):
    gp_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout
