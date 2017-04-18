import sys
sys.path.append('../../')

from  examples.gp_symbolic_regression import *

stdout = """0	33.4701298701	168.029018014	331.495213561
1	33.4701298701	123.035491283	31.7365717269
2	32.5	111.612206232	31.5742607582
3	32.5	102.507440224	24.5264979087
4	32.5	99.6274136033	45.0453687545
5	32.5	97.1714679473	57.9209802833
6	32.5	91.2474479828	55.6862773802
7	32.5	90.1140388835	89.8066443715
8	20.3571428571	78.3491400877	61.9190888051
9	20.3571428571	78.8836674333	120.362466574
10	20.3571428571	75.6879568355	96.0608389304
11	20.3571428571	71.5994753743	94.8459283234
12	20.3571428571	71.9815126381	126.6709029
13	20.3571428571	70.4340112283	120.364006141
14	20.3571428571	73.4109791028	143.992826247
15	0.0	68.6348696974	106.38294701
16	0.0	86.9964427507	381.136323331
17	0.0	72.2200064143	137.357846945
18	0.0	67.6290157472	101.481036957
19	0.0	72.7136442854	118.089497129
fitness: 0.0	genome: [4 9 4 1 6 6 3 9 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 """

def test_gp_with_elitism(capfd):
    gp_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout
