import sys
sys.path.append('../../')

from  examples.ga_sphere import *

stdout = """0	167.494019384	255.105414164	36.7647086254
1	155.867199569	230.002133389	33.7283523862
2	126.770683705	210.017901822	30.9938404139
3	126.770683705	194.008348719	29.8863856035
4	111.216784994	175.684589142	27.1451587218
5	94.1434173836	159.92434285	29.2453258306
6	86.1047171409	147.409396686	27.3067388262
7	82.8573214623	135.293147905	24.6877265803
8	76.3918723098	122.190875282	24.3556103848
9	58.5986349846	106.055169814	23.3519688567
10	47.1266028409	92.9430527188	18.2073704556
11	47.1266028409	81.4087654719	15.8652606292
12	37.8086403776	74.7437128381	14.76229207
13	33.5939266908	69.0963113693	15.7190679505
14	33.5939266908	61.9124728615	12.9197252953
15	31.0081923611	56.2314870124	11.7390642693
16	29.0475604046	51.6360343545	11.4376068048
17	22.3642960335	45.5689957244	9.42194725562
18	22.3642960335	42.0207146228	10.4016165791
19	20.090737207	37.718173719	8.40534074308
20	18.2899144853	36.077340326	8.88146599592
21	17.2317053479	32.1529678731	7.12350370182
22	17.2317053479	31.4144782315	8.41406530718
23	15.8591274399	29.7500946004	9.41406782723
24	15.8395199106	27.8898376807	8.09839712187
25	15.8395199106	26.536616113	8.12057889969
26	14.2113551811	24.1169430252	7.17851942317
27	12.3249583875	21.8282665062	5.79137544874
28	12.3249583875	21.9877144079	6.98043736465
29	12.3249583875	19.7604496035	5.56198792055
30	11.9531196391	18.9094977337	5.79418674929
31	11.9531196391	17.8919266431	6.45160600593
32	11.6551181924	17.3631434735	5.87911315218
33	10.0835062532	17.3241429601	7.64834857042
34	10.0835062532	15.1352270661	3.77382298697
35	10.0835062532	15.9672117832	6.45533472107
36	9.43223676527	14.6674837119	6.51146126756
37	9.09504908437	13.211539659	5.11233919873
38	8.70647822577	13.1613267332	5.56981245065
39	8.70647822577	13.8958154251	7.54633468326
40	8.39220181153	13.0984967359	6.32658719358
41	8.18707865523	12.7177512935	6.96808298481
42	7.74843085114	13.4242744932	8.11101061337
43	7.72600963615	12.5416871453	8.66676972669
44	7.25116763544	11.8778470908	6.08885596927
45	6.97228474675	11.1593289624	5.23357877012
46	6.97228474675	11.2985292542	5.8743811616
47	6.97228474675	11.1605812936	5.90054706559
48	6.83077928198	11.3909192311	6.21500265915
49	6.83077928198	10.3495933522	5.31156124835
50	6.83077928198	11.2099614358	6.17712673302
51	6.5362224642	9.97279186059	4.71768815255
52	6.5362224642	10.2522066753	6.56741819982
53	6.07592519305	9.67657259247	5.89924258976
54	5.80059832207	11.3823147424	8.81904500246
55	5.80059832207	10.4963166901	7.29263727944
56	5.7226635269	10.0027979444	6.24139628939
57	5.54284026606	9.91413677308	7.16686634817
58	5.54284026606	10.6353754849	7.41570150566
59	5.26202637361	10.5384434287	7.52863500257
60	5.19972192813	10.0855395019	7.25512268896
61	5.19972192813	10.1810357351	7.16373438129
62	5.11930092036	9.01717348424	5.77742705605
63	4.86661016356	9.25670599798	7.82479583443
64	4.86661016356	9.82116005387	7.09424000642
65	4.85793601975	9.08642233324	7.11246139515
66	4.78455760361	8.59243014885	7.01373032404
67	4.26930976447	7.54454594059	5.54759354814
68	4.26930976447	9.24869159401	7.56762324752
69	4.09567731861	9.32655992407	7.36631982152
70	3.74908452862	9.68783351238	8.52864747395
71	3.72238303838	7.11031286177	4.80834644731
72	3.37366286114	7.62098288547	7.18905764634
73	3.37366286114	8.91229043505	6.60067552383
74	3.00851468645	7.2030679602	5.39777637742
75	3.00851468645	7.79008500377	6.90354847429
76	3.00851468645	7.53594914741	5.99092063721
77	2.6824608941	7.45695657936	7.20449865651
78	2.57820712316	6.64469734619	5.70119700299
79	2.2227813084	6.33427217005	5.3284641494
80	2.2227813084	5.57938406857	4.78911892942
81	1.97215732221	6.21415061162	6.14491968696
82	1.97215732221	5.64021749194	6.36221221788
83	1.83949305202	5.70894334934	5.30065251386
84	1.83949305202	4.36102606258	5.59001359226
85	1.83949305202	5.1472140929	5.87827165447
86	1.83949305202	5.79895924268	7.39711187947
87	1.69630843971	4.80398278421	5.69438916269
88	1.57854402536	4.87947817415	6.0444483492
89	1.5532720471	4.5752660992	5.8828414425
90	1.5532720471	4.17899158803	4.73286290005
91	1.54010019141	4.79000211453	7.21121420129
92	1.54010019141	4.97872223016	6.20987492937
93	1.54010019141	5.62051325954	7.3966158029
94	1.49558942043	5.02041994974	6.57593360503
95	1.49558942043	4.09618890716	5.94025938402
96	1.48184359333	4.02416611554	5.4357310447
97	1.29482287034	5.03518483499	6.44102215866
98	1.29482287034	4.86460350728	6.66044369231
99	1.29482287034	4.98543028351	6.13241199013
100	1.29482287034	4.41333509744	6.02608342348
101	1.21791029687	4.07388259723	5.67365062941
102	1.21791029687	3.68571229088	4.88803689311
103	1.19205722325	3.94546658729	5.94847672552
104	1.17946646293	5.69768524681	7.50254964251
105	1.17946646293	4.67927403155	6.32787809001
106	1.16319927799	3.76498938784	5.61000157177
107	1.16319927799	3.63484534273	5.80027791309
108	1.15778159238	4.48312509274	6.75705047611
109	1.06634003235	3.90558405601	5.80874301987
110	1.06634003235	2.98497856573	4.04343913984
111	1.05316817666	4.20103453414	7.08883607323
112	1.05007284741	2.55754039903	4.3714260935
113	1.05007284741	3.08465723774	5.07243252317
114	1.02830142197	2.78014955713	4.29370544154
115	0.994025791094	2.54219272232	4.5213117022
116	0.973657338101	3.08422712414	5.35013850641
117	0.973657338101	2.81243558266	4.20976650257
118	0.973657338101	3.68244747143	5.57565117437
119	0.958350728565	3.26417717518	4.90210484191
120	0.958350728565	4.73565540336	7.65667322839
121	0.957390153161	3.80621010973	6.28768940394
122	0.957390153161	4.0302995471	6.40201104004
123	0.953971564747	3.6588475372	5.9891902527
124	0.914456089426	3.30861984972	5.51881057684
125	0.889589334727	4.05262225037	6.7151354867
126	0.889589334727	3.913866758	5.5739316867
127	0.889589334727	4.04340508433	6.10707103373
128	0.865802293321	4.36038581634	7.02934042657
129	0.852953696794	4.3105618264	6.55100462206
130	0.788606932376	5.8001642489	8.52840287736
131	0.788606932376	4.55035045537	8.48309075498
132	0.788606932376	3.14045968965	5.69235435086
133	0.676216783572	3.47452295366	6.09203184816
134	0.676216783572	4.17834476672	7.38313176335
135	0.676216783572	4.04942885245	6.44346907107
136	0.676216783572	3.0286114314	5.64265808735
137	0.676216783572	3.50891018772	6.04195808403
138	0.671629834673	3.49555625304	6.50830682363
139	0.614003962774	3.35928342883	5.81276009677
140	0.614003962774	3.75123769348	6.73734684833
141	0.614003962774	4.23605970513	7.07707575059
142	0.614003962774	4.14435891081	7.01801345243
143	0.61058537436	4.02816340799	6.18368334907
144	0.61058537436	3.56498398338	6.07698621132
145	0.61058537436	2.89802546412	5.3165505885
146	0.61058537436	2.90888108021	4.92046053645
147	0.605042469398	2.65313886142	4.94224063855
148	0.605042469398	2.92702048693	5.52709368817
149	0.605042469398	4.15101241362	6.2643942051
150	0.601623880985	4.12956384544	6.45184405817
151	0.601623880985	3.38367646716	5.78587488257
152	0.601623880985	3.95048246041	6.78922284982
153	0.601220145534	4.8232501215	7.55406095291
154	0.601220145534	3.79984669036	6.03811029981
155	0.601220145534	4.19154066422	7.17218048265
156	0.592258652159	2.74885670894	5.43734210122
157	0.592258652159	4.27013585217	7.41761531471
158	0.592258652159	3.60853235693	6.3975632641
159	0.586895768693	4.58767217503	8.53594709885
160	0.586895768693	2.80227931803	4.89570743904
161	0.586895768693	3.53928464555	6.18285763783
162	0.577530539867	3.58782915238	5.73115586474
163	0.577530539867	4.68439041067	7.73627796522
164	0.542593719088	5.60326556522	8.19076357465
165	0.542593719088	5.5696513658	7.99364844518
166	0.531343947885	3.42529470237	5.09197321756
167	0.516615835593	4.59781678521	7.57646501677
168	0.516615835593	3.31311685381	4.71374710099
169	0.516615835593	3.71132427794	6.28117767931
170	0.516615835593	3.87697510843	6.24514649798
171	0.516615835593	3.56180859824	6.66482531477
172	0.509358222717	3.48150793054	6.7049828649
173	0.509358222717	2.95420585915	5.63448442476
174	0.509358222717	4.35309729064	7.62197750587
175	0.406270918051	3.87664066059	6.10025432113
176	0.406270918051	3.711813144	6.34303889253
177	0.406270918051	4.19577909353	7.0480319534
178	0.404386375673	4.44224156991	6.74207111398
179	0.397128762797	4.50518537085	7.42966855099
180	0.397128762797	4.06664650909	7.61497220282
181	0.397128762797	2.74938549496	5.4516709126
182	0.397128762797	3.2617959871	6.80280665105
183	0.397128762797	3.47751637291	6.3440319712
184	0.397128762797	3.99893946125	6.89814522806
185	0.397128762797	4.12618882626	6.46212999878
186	0.397128762797	4.07569972604	7.13273278028
187	0.386560456711	5.06975688271	7.87080909121
188	0.386560456711	3.14267449465	6.26303592016
189	0.386560456711	3.81173354702	6.97815834904
190	0.386560456711	3.98084173665	6.261334235
191	0.386560456711	3.00101038137	5.16600181837
192	0.386560456711	3.31197955354	6.21252808428
193	0.386560456711	4.27002253059	6.88671619431
194	0.386560456711	3.2736714672	5.81756448615
195	0.386560456711	2.83890606366	5.52340317942
196	0.386560456711	2.61610003008	4.814025816
197	0.386560456711	3.09579643085	6.78955037591
198	0.364919045479	2.97036448231	5.98609140574
199	0.364919045479	3.65784656916	6.5477630337
fitness: 0.364919045479	genotype: [  3.96592965e-02  -2.22956646e-01  -2.15501682e-02   3.07089867e-02
  -1.13568096e-01  -5.71659323e-02   7.71501247e-02  -1.90271445e-04
  -1.04464237e-01  -2.78778014e-02  -8.38412488e-02   5.52440926e-02
   5.88867888e-02  -2.36277007e-02  -2.77398933e-02   1.40725891e-02
   1.66983852e-01   1.28234760e-01   5.14333949e-02   1.23637775e-01
   2.48784736e-01   1.06925504e-02   2.40216552e-01   9.43578344e-02
   7.10057785e-02   1.85360030e-01  -1.04924410e-01  -4.59915535e-02
  -1.23718139e-01  -6.81237290e-02]
"""

def test_generational_with_elitism(capfd):
    generational_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout
