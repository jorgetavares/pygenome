from examples.ga_onemax import *

stdout1 = """0	0.03125	0.0407813020097	0.0055614551069
1	0.0294117647059	0.0380918616101	0.00450622600605
2	0.0277777777778	0.0354722826392	0.0037863687237
3	0.0277777777778	0.0338362202776	0.00379265767648
4	0.0277777777778	0.0315314487988	0.00246353628866
5	0.0238095238095	0.0301390249535	0.00243139455637
6	0.0238095238095	0.028943889401	0.00265769830081
7	0.0238095238095	0.0274036610815	0.00182631053989
8	0.0232558139535	0.0264441876819	0.00153496532773
9	0.0227272727273	0.0260128701016	0.00160853816813
10	0.0222222222222	0.0253143903549	0.00130642239486
11	0.0222222222222	0.0247527987488	0.00141703794024
12	0.0212765957447	0.0240260411066	0.00134274189815
13	0.0212765957447	0.0233757082989	0.00120068859079
14	0.0208333333333	0.0228546785563	0.00101858633894
15	0.02	0.022430398718	0.000870433399703
16	0.02	0.02196035083	0.000894969777647
17	0.02	0.0216537169847	0.000710159049295
18	0.02	0.0213578600504	0.000618545520307
19	0.02	0.0211115618416	0.000639321623828
fitness: 0.02	genotype: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1]
"""

stdout2 = """0	0.03125	0.0407813020097	0.0055614551069
1	0.0294117647059	0.0380595340239	0.00454344387043
2	0.0277777777778	0.035407148563	0.00400372903938
3	0.0277777777778	0.033389549421	0.00321945068766
4	0.025	0.0317596301113	0.00241383035772
5	0.0243902439024	0.0304658308858	0.00245067600853
6	0.0243902439024	0.0297481699486	0.00224093114127
7	0.0232558139535	0.0287829398706	0.0021429324656
8	0.0232558139535	0.0275022496355	0.00191233976138
9	0.0232558139535	0.026455506916	0.00182884732022
10	0.0222222222222	0.0253671004707	0.00151992811902
11	0.0222222222222	0.0246190726278	0.00134999235453
12	0.0217391304348	0.0240653228253	0.00117297763618
13	0.0212765957447	0.0234613321829	0.00106703169858
14	0.0212765957447	0.0228950762862	0.000939610529858
15	0.0208333333333	0.0225220165679	0.000826833817142
16	0.0204081632653	0.0221363399766	0.00078043853362
17	0.02	0.0218176099515	0.000832777948979
18	0.02	0.0214730956676	0.00071787980309
19	0.02	0.0212076047124	0.000674141765573
fitness: 0.02	genotype: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1]
"""

stdout3 = """0	0.03125	0.0407813020097	0.0055614551069
1	0.03125	0.0407659173943	0.00556572141959
2	0.03125	0.0406220427504	0.00550716456476
3	0.03125	0.0406271957295	0.00552550438654
4	0.03125	0.0405725603488	0.0054984646903
5	0.03125	0.0403409049214	0.00539038407237
6	0.03125	0.0402046536026	0.00541977894116
7	0.03125	0.0401373932768	0.0054623937675
8	0.03125	0.0400744303138	0.00533801740002
9	0.03125	0.0400845884826	0.00531976268578
10	0.03125	0.0401068354236	0.00529156312546
11	0.03125	0.0400226603394	0.00527274230941
12	0.03125	0.0400522899691	0.0052641997698
13	0.03125	0.0399878777308	0.00526129214322
14	0.03125	0.039846640657	0.00531620308308
15	0.03125	0.0398888501078	0.00529946693157
16	0.03125	0.0397690225216	0.00541019444543
17	0.03125	0.0396842930855	0.00543528026051
18	0.03125	0.039515774567	0.00520608147101
19	0.03125	0.039352360395	0.00516121235895
20	0.03125	0.0394384444897	0.00513823519102
21	0.03125	0.0392122540135	0.00506918157756
22	0.03125	0.0391049846323	0.00506621901174
23	0.03125	0.039126629654	0.00509782600988
24	0.03125	0.0391249762148	0.00504608772719
25	0.03125	0.0390786799185	0.00504379331456
26	0.03125	0.0390466286365	0.00503742549295
27	0.03125	0.0390786799185	0.00504379331456
28	0.03125	0.0389681503619	0.00496252839987
29	0.03125	0.0390079381603	0.00494231567345
30	0.03125	0.0389511456774	0.00488617726648
31	0.03125	0.0389576875453	0.00495978158922
32	0.03125	0.038884355036	0.00501690912459
33	0.03125	0.0387625159555	0.00506275794984
34	0.0294117647059	0.0387118060164	0.00513031663951
35	0.0294117647059	0.0387737632322	0.00509184828717
36	0.0294117647059	0.0387641987226	0.0051088177607
37	0.0294117647059	0.0385806766181	0.00517653770352
38	0.0294117647059	0.0385760446909	0.00516256817574
39	0.0294117647059	0.0384000993303	0.00491179256108
40	0.0294117647059	0.0384359417987	0.00498119789458
41	0.0294117647059	0.0385350135751	0.00491732298477
42	0.0294117647059	0.0385510715912	0.004878023382
43	0.0294117647059	0.0385082144483	0.00488392823313
44	0.0294117647059	0.0384861531763	0.00495451256888
45	0.0294117647059	0.0383247775149	0.00463031576246
46	0.0294117647059	0.038350516782	0.00461749193524
47	0.0294117647059	0.0383522500424	0.00462288975441
48	0.0294117647059	0.0382959889016	0.00461175402247
49	0.0294117647059	0.0383351830408	0.00459520117791
50	0.0294117647059	0.0383449873818	0.00463114541678
51	0.0294117647059	0.0382493387775	0.00465393947571
52	0.0294117647059	0.0382064816346	0.00465735254217
53	0.0294117647059	0.0381156143699	0.00463845923586
54	0.0294117647059	0.0380422818605	0.00468633600549
55	0.0294117647059	0.0380540159448	0.00472897874711
56	0.0294117647059	0.0379536814966	0.0046656238405
57	0.0294117647059	0.0379049635479	0.00456352756167
58	0.0294117647059	0.0379467608311	0.00451256664974
59	0.0294117647059	0.0378634274978	0.00452004131202
60	0.0294117647059	0.0378729920073	0.00450276630527
61	0.0294117647059	0.0377247934317	0.00457719773318
62	0.0294117647059	0.0375611029555	0.00459051501874
63	0.0294117647059	0.0375312087756	0.00458194511291
64	0.0285714285714	0.0374145014089	0.00464957739777
65	0.0285714285714	0.0374597802042	0.00466464184004
66	0.0285714285714	0.037440549435	0.00469954230283
67	0.0294117647059	0.0374764253238	0.00468220417022
68	0.0294117647059	0.0374327471629	0.00467600604283
69	0.0294117647059	0.0374077494604	0.00468562811634
70	0.0294117647059	0.0373172517229	0.00475133391939
71	0.0294117647059	0.0371447649504	0.00464453250789
72	0.0294117647059	0.0372306092137	0.00460630503674
73	0.0294117647059	0.0371933323452	0.00456849074203
74	0.0294117647059	0.0370724990119	0.00457037180035
75	0.0294117647059	0.0372588866982	0.00463564051399
76	0.0294117647059	0.0373779965644	0.00457285182912
77	0.0294117647059	0.0373495335663	0.00461331065151
78	0.0294117647059	0.0374115683058	0.00458605733025
79	0.0294117647059	0.0373158191475	0.00457549606542
80	0.0294117647059	0.0372861895179	0.00456760498028
81	0.0294117647059	0.0371575773893	0.00446634515156
82	0.0294117647059	0.0371841376745	0.00446247417508
83	0.0294117647059	0.0372488961449	0.00440237585026
84	0.0294117647059	0.0371422744557	0.00435184202632
85	0.0294117647059	0.0369771703718	0.00417687382414
86	0.0294117647059	0.0369112452699	0.00418362269464
87	0.0294117647059	0.0368450448037	0.00424055860449
88	0.0294117647059	0.0367962646078	0.00430644517518
89	0.0294117647059	0.0368085798788	0.004301569721
90	0.0294117647059	0.0367852717204	0.00431466312322
91	0.0294117647059	0.0366909093947	0.00436321528596
92	0.0294117647059	0.0367041369079	0.00436223942143
93	0.0294117647059	0.0366487440974	0.00429285971394
94	0.0294117647059	0.0366472506685	0.0043439846922
95	0.0294117647059	0.0364240363828	0.00420293196706
96	0.0294117647059	0.036346617028	0.00420765034446
97	0.0294117647059	0.0361374453675	0.0041893800435
98	0.0294117647059	0.0361018219381	0.00420205885431
99	0.0294117647059	0.0360297065535	0.0042227736296
100	0.0294117647059	0.0359238864477	0.00406049727444
101	0.0294117647059	0.0359521799932	0.00407146251789
102	0.0294117647059	0.0359281997918	0.00407834648928
103	0.0294117647059	0.0358530866697	0.00413392632262
104	0.0294117647059	0.0358450513728	0.00409428679465
105	0.0294117647059	0.0358803766966	0.00403267246207
106	0.0294117647059	0.0359131531943	0.00406374241496
107	0.0294117647059	0.0357711077397	0.0039750338135
108	0.0294117647059	0.0357683569783	0.00396253691982
109	0.0294117647059	0.0356733967661	0.00393371788136
110	0.0294117647059	0.0356221147149	0.00393046156945
111	0.0294117647059	0.0355814146742	0.00391744792455
112	0.0294117647059	0.0356137422604	0.00389484429588
113	0.0294117647059	0.0355419031799	0.00384851445846
114	0.0294117647059	0.0356152356893	0.00383814285161
115	0.0294117647059	0.0355626857179	0.00388053919793
116	0.0294117647059	0.0356205560883	0.00385888326125
117	0.0294117647059	0.0355807682899	0.00384988713827
118	0.0294117647059	0.0354710213488	0.00386007686302
119	0.0294117647059	0.0354366822684	0.00381722297753
120	0.0294117647059	0.0353712060779	0.00385648271004
121	0.0294117647059	0.0352858631801	0.00386172816807
122	0.0294117647059	0.0352757825349	0.00387092354526
123	0.0294117647059	0.0352809373746	0.00384358262514
124	0.0285714285714	0.0352593065	0.00385349215061
125	0.0277777777778	0.0352245842778	0.00390476419982
126	0.0277777777778	0.0350222738169	0.00383235376111
127	0.0277777777778	0.0348271441521	0.00355983591502
128	0.0277777777778	0.0347286444878	0.00360166243916
129	0.0277777777778	0.0347215638777	0.00350437216356
130	0.0277777777778	0.0346531114968	0.00352079792056
131	0.0277777777778	0.034629301973	0.00352159173397
132	0.0277777777778	0.0345945797508	0.00357150384681
133	0.0277777777778	0.0345384426407	0.00358714307784
134	0.0277777777778	0.0345839977402	0.00357854947167
135	0.0277777777778	0.0345549777011	0.00360644599842
136	0.0277777777778	0.0342354589845	0.00323394946738
137	0.0277777777778	0.0342239647316	0.00323509254513
138	0.0277777777778	0.0342094942883	0.00327386960546
139	0.0277777777778	0.0341940321493	0.00325411656013
140	0.0277777777778	0.0341516239179	0.00327452428579
141	0.0277777777778	0.0342087126625	0.00321724072036
142	0.0277777777778	0.0342087126625	0.00321724072036
143	0.0277777777778	0.0341913207802	0.00318475243626
144	0.0277777777778	0.0340368641257	0.00326439112698
145	0.0277777777778	0.0339027140017	0.00321807484556
146	0.0277777777778	0.033818057917	0.00324573956032
147	0.0277777777778	0.0337777938229	0.00326868684448
148	0.0277777777778	0.0337474907926	0.00328666392829
149	0.0277777777778	0.0337353244967	0.0032830776386
150	0.0277777777778	0.0335041617975	0.00315062008131
151	0.0277777777778	0.0334833284641	0.00315855876935
152	0.0277777777778	0.0335265687488	0.00309992460878
153	0.0277777777778	0.0334615434962	0.00316030081197
154	0.0277777777778	0.0334209142023	0.00317883675768
155	0.0277777777778	0.0333668016481	0.00318538717186
156	0.0277777777778	0.0333924512042	0.00320178131556
157	0.027027027027	0.0334241593829	0.00319091133628
158	0.027027027027	0.0333722814546	0.00320922169665
159	0.027027027027	0.0333008528832	0.00323568136595
160	0.027027027027	0.0332169943397	0.00325540390129
161	0.0277777777778	0.0332685631499	0.0031931312795
162	0.0277777777778	0.0331622976552	0.00325887197309
163	0.0277777777778	0.0331767133471	0.00333213897336
164	0.0277777777778	0.0330957003776	0.00336208463916
165	0.0277777777778	0.0330676316721	0.00338171217065
166	0.0277777777778	0.0330777123173	0.00337777911157
167	0.0277777777778	0.0329720522217	0.00333538995836
168	0.0277777777778	0.0329006236503	0.00335233956713
169	0.0277777777778	0.0328039425536	0.00339588667117
170	0.0277777777778	0.0327621452704	0.00340068532175
171	0.0277777777778	0.0327603052382	0.00339295706616
172	0.0277777777778	0.0326876328126	0.00334521636578
173	0.0277777777778	0.0325664847727	0.00331382976599
174	0.027027027027	0.0325026801449	0.0033544687941
175	0.027027027027	0.0323940998354	0.00333273008703
176	0.027027027027	0.0323570627983	0.00330135896684
177	0.0277777777778	0.0323920571559	0.0032733578152
178	0.0277777777778	0.0323320824084	0.00332872687708
179	0.0277777777778	0.0322147058026	0.00324464624573
180	0.0277777777778	0.0322134228114	0.00323851735812
181	0.0277777777778	0.0320760273514	0.00326035636594
182	0.027027027027	0.0320144072898	0.00325505254887
183	0.027027027027	0.0319467286054	0.00327234966231
184	0.027027027027	0.0319093529514	0.00328746246683
185	0.027027027027	0.0319093529514	0.00328746246683
186	0.027027027027	0.032001945544	0.00330016187319
187	0.027027027027	0.0319258256453	0.00331843751825
188	0.027027027027	0.0318805559245	0.00334226105485
189	0.027027027027	0.0319619634875	0.0033456141461
190	0.027027027027	0.0319227099482	0.00334636806319
191	0.027027027027	0.0319871781598	0.00331742402805
192	0.027027027027	0.0320164148544	0.00331161418507
193	0.027027027027	0.0319608592989	0.00333556888564
194	0.027027027027	0.0318649972046	0.00311146352897
195	0.027027027027	0.0317904777417	0.0030888021594
196	0.027027027027	0.0318179502692	0.00313544444399
197	0.027027027027	0.0317773959161	0.00307859627228
198	0.027027027027	0.031758402615	0.0030841676145
199	0.027027027027	0.0317028470594	0.00310526216733
fitness: 0.027027027027	genotype: [1 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 0 0
 0 1 1 1 1 1 0 1 1 0 0 1 1]
"""

stdout4 = """0	0.03125	0.0407813020097	0.0055614551069
1	0.03125	0.0407335898081	0.0056113238941
2	0.03125	0.0405318447939	0.00561889658407
3	0.03125	0.0405294321203	0.0057074300332
4	0.03125	0.0403188513205	0.00555715570344
5	0.03125	0.040176813266	0.00555018239064
6	0.03125	0.0400723702951	0.00565983496909
7	0.03125	0.0397406170495	0.00567552626167
8	0.03125	0.0396199062938	0.00568178587844
9	0.03125	0.0394976042503	0.00573239516276
10	0.03125	0.039577401104	0.0057396592945
11	0.0285714285714	0.0394184725326	0.00588830862201
12	0.0285714285714	0.0394638022029	0.00587549024626
13	0.0285714285714	0.0392980672133	0.00595596723957
14	0.0285714285714	0.0391270507298	0.00610152645969
15	0.0285714285714	0.0388853275671	0.00624167731371
16	0.0285714285714	0.038668706673	0.00632051218983
17	0.0285714285714	0.038567648472	0.00639373332562
18	0.0285714285714	0.0383116771233	0.00654060158892
19	0.0285714285714	0.0382270210386	0.00661095907092
20	0.0285714285714	0.037923786049	0.00670319168539
21	0.0285714285714	0.0377105992358	0.00682573538465
22	0.0285714285714	0.0376520968835	0.00687964614961
23	0.0285714285714	0.0375605217919	0.00680693000063
24	0.0285714285714	0.0372530984033	0.00661624489732
25	0.0285714285714	0.0370714726217	0.00670093541862
26	0.0277777777778	0.0368947690262	0.00681638917178
27	0.0277777777778	0.0367933276834	0.00673774032456
28	0.0277777777778	0.036565926366	0.00671879565136
29	0.0277777777778	0.0363629998386	0.00671820660982
30	0.0277777777778	0.0361597495758	0.00678462419596
31	0.0277777777778	0.0360010674611	0.00679273624476
32	0.0263157894737	0.0358570540449	0.00688600819611
33	0.0263157894737	0.035715119347	0.00698096660186
34	0.0263157894737	0.0355820814434	0.00697567127266
35	0.0263157894737	0.0355232696412	0.00702983056422
36	0.0263157894737	0.0353526305209	0.00710900624447
37	0.0263157894737	0.0352590401521	0.00716713853184
38	0.0263157894737	0.0350031504278	0.00726173343824
39	0.0263157894737	0.0350878065124	0.00723555819287
40	0.0263157894737	0.0350603196623	0.00725335416603
41	0.0263157894737	0.0349749637996	0.0072489778619
42	0.0263157894737	0.0348118650805	0.00696666817941
43	0.0263157894737	0.0347328933457	0.00699310657924
44	0.0263157894737	0.0345763387648	0.0070796600767
45	0.025641025641	0.0344595347355	0.0070966355864
46	0.025641025641	0.034242018903	0.00711512056942
47	0.025641025641	0.0340000928013	0.00716816580159
48	0.025641025641	0.0336707427844	0.00711148149312
49	0.025	0.0334364123001	0.00718945066675
50	0.025	0.033447997636	0.00718747787141
51	0.025	0.0332287118441	0.00724924331215
52	0.025	0.0330933317272	0.00725059006156
53	0.025	0.0329527267181	0.00730886134113
54	0.025	0.0328793050418	0.00736639598827
55	0.025	0.0327844774556	0.00740607328012
56	0.025	0.0327926870822	0.00740120105533
57	0.025	0.0326800198581	0.00738431363982
58	0.025	0.0323000872197	0.00724585027197
59	0.0243902439024	0.0321439896587	0.00724643011145
60	0.0243902439024	0.0320158660044	0.00726814250372
61	0.0243902439024	0.0318834854522	0.00733690135936
62	0.0243902439024	0.0318168809962	0.00737690291083
63	0.0243902439024	0.0314620932816	0.00717743738592
64	0.0243902439024	0.0309626072359	0.00682354878219
65	0.0243902439024	0.0303097164097	0.00613846819504
66	0.0243902439024	0.0299898536665	0.00562021355961
67	0.0243902439024	0.029909236022	0.00565701688686
68	0.0243902439024	0.0297141977297	0.00550742023601
69	0.0243902439024	0.0296412098519	0.00554076046008
70	0.0243902439024	0.0295909946438	0.00556531264999
71	0.0243902439024	0.0295402847047	0.00554356934474
72	0.0238095238095	0.0294674131318	0.00557405836839
73	0.0238095238095	0.029395570937	0.00556082224202
74	0.0238095238095	0.0292635517128	0.00550738816979
75	0.0238095238095	0.0290572025065	0.00540440001722
76	0.0238095238095	0.028922710872	0.00539078117314
77	0.0238095238095	0.0289379980737	0.0053868832612
78	0.0238095238095	0.0288807225319	0.00540873210938
79	0.0238095238095	0.02888038515	0.00540813338838
80	0.0238095238095	0.0286927799624	0.0052747239615
81	0.0238095238095	0.0284665894862	0.00515413522599
82	0.0238095238095	0.0283781575929	0.00517392101898
83	0.0238095238095	0.0282936721858	0.00517885202489
84	0.0238095238095	0.0281222373405	0.00497234257179
85	0.0238095238095	0.0280682080365	0.00499411336305
86	0.0238095238095	0.0280682080365	0.00499411336305
87	0.0238095238095	0.027946740415	0.00491842700352
88	0.0238095238095	0.0278866675872	0.0049005050364
89	0.0238095238095	0.0279223818729	0.00489234448706
90	0.0238095238095	0.0279037294727	0.00490353688633
91	0.0238095238095	0.0279295321014	0.00490905901247
92	0.0238095238095	0.0278447093125	0.00490580576841
93	0.0238095238095	0.0278572171299	0.00489857149548
94	0.0238095238095	0.0277947844645	0.00491571955458
95	0.0238095238095	0.0276621820819	0.00480937926453
96	0.0238095238095	0.0275416845239	0.00482841922029
97	0.0238095238095	0.0274990109635	0.00481039987069
98	0.0238095238095	0.0273582980179	0.00469201673247
99	0.0238095238095	0.0273022082743	0.00467866559603
100	0.0238095238095	0.0271624357661	0.00455293227425
101	0.0238095238095	0.0271512331233	0.00456194644299
102	0.0232558139535	0.026918788874	0.00428924290421
103	0.0232558139535	0.026900743958	0.00429911458798
104	0.0232558139535	0.0267935658792	0.00411819410199
105	0.0232558139535	0.0267375434702	0.00412030090784
106	0.0232558139535	0.0267710020381	0.00416363146773
107	0.0232558139535	0.0267213472925	0.00416303393603
108	0.0232558139535	0.0267158101939	0.00416726924333
109	0.0232558139535	0.0265768078041	0.00417039084575
110	0.0232558139535	0.0265609347882	0.00416540903332
111	0.0232558139535	0.0265630184858	0.00417060368392
112	0.0232558139535	0.0266104938813	0.00421484207235
113	0.0232558139535	0.0266288088996	0.00420660226409
114	0.0232558139535	0.0265841592619	0.0041969238651
115	0.0232558139535	0.0265853698017	0.00419997458126
116	0.0232558139535	0.0264722229926	0.00421576952647
117	0.0232558139535	0.0264535279993	0.00422212577858
118	0.0232558139535	0.0263203513411	0.00409035803251
119	0.0232558139535	0.0263026393783	0.0040990381548
120	0.0232558139535	0.0262291751089	0.00410342164579
121	0.0232558139535	0.026271679626	0.00408711272348
122	0.0232558139535	0.0262649319876	0.00408759130436
123	0.0232558139535	0.0262463266093	0.00409370889282
124	0.0232558139535	0.0262359450826	0.00410312992275
125	0.0232558139535	0.026190075367	0.00410892875105
126	0.0232558139535	0.0261356234949	0.00412908654297
127	0.0232558139535	0.0262059057898	0.00411517520351
128	0.0232558139535	0.0261933083575	0.00411826533423
129	0.0232558139535	0.0258390459102	0.00347734143293
130	0.0232558139535	0.025808446155	0.00348646312548
131	0.0232558139535	0.0257313215869	0.00349375436407
132	0.0232558139535	0.0257836536007	0.00349249937216
133	0.0232558139535	0.0257284107406	0.00348347366855
134	0.0232558139535	0.0256757724638	0.00350598644735
135	0.0232558139535	0.0255411570791	0.00326245739128
136	0.0232558139535	0.0254959374409	0.00326248166658
137	0.0232558139535	0.0254766819852	0.00326326827899
138	0.0232558139535	0.025450604514	0.00326227942895
139	0.0232558139535	0.0254698599697	0.00326164653264
140	0.0232558139535	0.0254640527688	0.00326407963037
141	0.0232558139535	0.0254527084693	0.00326975835072
142	0.0227272727273	0.0253426143202	0.00326212802481
143	0.0227272727273	0.0253109610767	0.00326282511233
144	0.0227272727273	0.0252687082459	0.00327151082501
145	0.0227272727273	0.0251972524449	0.00319915986612
146	0.0227272727273	0.0251570223025	0.00321373699607
147	0.0227272727273	0.0251461997917	0.00322007261739
148	0.0227272727273	0.0251143897345	0.00321756319799
149	0.0227272727273	0.025078504567	0.00322396880421
150	0.0227272727273	0.0250668514914	0.00322720042144
151	0.0227272727273	0.0249131020151	0.0031401751833
152	0.0227272727273	0.0249186391137	0.00313773522364
153	0.0227272727273	0.0248711865662	0.00314531651468
154	0.0227272727273	0.0247821452526	0.00313039276923
155	0.0227272727273	0.0247517971837	0.00313102304395
156	0.0227272727273	0.0247343553232	0.00313444806866
157	0.0227272727273	0.0247649550784	0.00313480092573
158	0.0227272727273	0.0247316956549	0.00314739661827
159	0.0227272727273	0.0246624315856	0.00313460088194
160	0.0222222222222	0.0246159588146	0.00314138257457
161	0.0222222222222	0.0245192777179	0.0031054155193
162	0.0222222222222	0.0245028829133	0.00311123536366
163	0.0222222222222	0.0244628876782	0.00311800046592
164	0.0222222222222	0.0244361569564	0.00312912889538
165	0.0222222222222	0.0244313581376	0.00313312977456
166	0.0222222222222	0.0244393372386	0.00314041214926
167	0.0222222222222	0.0244087374834	0.00313688518533
168	0.0222222222222	0.0243642135319	0.00314725365926
169	0.0222222222222	0.0243581159709	0.00314660657645
170	0.0222222222222	0.0243086579764	0.00316003120598
171	0.0222222222222	0.0242562585261	0.00313504764816
172	0.0222222222222	0.024229292897	0.00314362198895
173	0.0222222222222	0.0242081344689	0.00315174821802
174	0.0222222222222	0.0242028490566	0.00315378332571
175	0.0222222222222	0.0241490290028	0.0031473309832
176	0.0222222222222	0.0241545661013	0.00314624156175
177	0.0222222222222	0.0241283571682	0.00315651073465
178	0.0222222222222	0.0241389279927	0.00315268364361
179	0.0222222222222	0.0241235415703	0.00315989335658
180	0.0222222222222	0.0240944040412	0.00315920007551
181	0.0222222222222	0.023978242425	0.00312764579468
182	0.0222222222222	0.0239341050379	0.00313383395436
183	0.0222222222222	0.0238355140863	0.00309580561591
184	0.0222222222222	0.0237428844937	0.00305889746244
185	0.0222222222222	0.0237597491127	0.00305864553506
186	0.0222222222222	0.023684625406	0.00304897085581
187	0.0217391304348	0.0236745090758	0.00305285287304
188	0.0217391304348	0.0236371754303	0.00305935308227
189	0.0217391304348	0.0236373950175	0.00306004973154
190	0.0217391304348	0.0236116405787	0.0030676901382
191	0.0217391304348	0.0235675711216	0.00306824760095
192	0.0217391304348	0.023530994172	0.00307622597276
193	0.0217391304348	0.023530994172	0.00307622597276
194	0.0217391304348	0.0235206582546	0.00307886846146
195	0.0217391304348	0.0234859337623	0.00307094287886
196	0.0217391304348	0.0234652298285	0.00307566702128
197	0.0217391304348	0.0232720816293	0.00272300001115
198	0.0217391304348	0.0232311603553	0.00273225476308
199	0.0217391304348	0.0232362108604	0.00273085152445
fitness: 0.0217391304348	genotype: [1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 1 1 1 1 1]
"""


def test_generational_no_elitism(capfd):
    generational_no_elitism()
    out, err = capfd.readouterr()
    assert out == stdout1

def test_generational_with_elitism(capfd):
    generational_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout2

def test_steady_state_no_elitism(capfd):
    steady_state_no_elitism()
    out, err = capfd.readouterr()
    assert out == stdout3

def test_steady_state_with_elitism(capfd):
    steady_state_with_elitism()
    out, err = capfd.readouterr()
    assert out == stdout4