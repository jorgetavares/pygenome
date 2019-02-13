from examples.stgp_symbolic_regression import *

stdout = """0	42.55060553037077	146.3300529726899	109.00378461184702
1	38.98045012013038	134.69091025604652	27.744085415668522
2	33.64293454337553	129.95842826076603	38.9917684745024
3	33.64293454337553	122.88815785775573	33.86217179254338
4	33.64293454337553	114.05752911012638	29.98481586562082
5	33.64293454337553	109.06766651602446	37.52969896982413
6	33.64293454337553	105.19327373469845	38.61848079868762
7	33.64293454337553	99.64168453142848	42.580638115022275
8	33.64293454337553	100.70040783276056	83.15279290441762
9	33.64293454337553	94.06107034928812	63.28778103922939
10	33.64293454337553	92.99976286081747	79.04366695824943
11	33.64293454337553	112.24332670188561	789.0839868341842
12	33.64293454337553	88.97378818675321	78.14598064698387
13	31.560883666504754	84.90696580483392	63.101049212697006
14	31.560883666504754	87.83478270864732	107.2474334710058
15	30.80655316204825	85.78107118189044	76.03431972783373
16	30.418820174441255	86.7716652646136	70.91997431144989
17	30.418820174441255	89.50451724805819	151.77494931424053
18	30.418820174441255	83.06962681412388	54.73776491852288
19	30.297052841431757	88.421302048993	89.90380778342491
fitness: 30.297052841431757	genotype: [ 1  3  1  4  2 16 11  7  6  4  1  3  8 10 15  1 16 16 16  1 16 16 16  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
"""


def test_stgp_with_elitism(capfd):
    stgp_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout
