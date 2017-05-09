from  examples.stgp_symbolic_regression_ephemeral import *

stdout = """0	34.7797657585	1929.94989643	20588.4157684
1	32.5	230.593606801	1319.72073855
2	32.2158317278	203.27265759	1805.48101457
3	32.2158317278	176.203896381	1136.0759831
4	32.2158317278	153.707570015	608.0473584
5	27.9312403959	132.547345191	331.29159466
6	27.9312403959	222.087866048	1868.7545496
7	27.9312403959	157.777320503	755.733297434
8	27.9312403959	172.333797412	883.81258447
9	27.9312403959	194.072887473	1165.27597561
10	27.9312403959	364.59143761	4699.68197014
11	25.9312403959	638.314524201	13246.10976
12	25.9312403959	149.698134034	551.025282999
13	25.9312403959	369.165453865	5779.79648697
14	25.9312403959	153.933981481	763.176386596
15	25.9312403959	1977.61800596	58556.8905176
16	25.9312403959	174.262803229	993.237791238
17	25.9312403959	171.505568447	992.433438516
18	25.9312403959	320.329353207	6619.31484113
19	25.9312403959	128.221653301	579.890146997
fitness: 25.9312403959	genotype: [   6    3    1    3    2 8031    9    1    9    9    9    5    1    9    4
    1    9    9    5    5   13   33   15    1    9    9    9    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0]
"""

def test_stgp_with_elitism_ephemeral(capfd):
    stgp_with_elitism_ephemeral()
    out, err = capfd.readouterr()
    assert out == stdout