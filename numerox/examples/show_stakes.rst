Staking information
===================

Numerox provides information about the staking tournament::

    >>> import numerox as nx
    >>> nx.show_stakes()
    days    s    soc   cumsum  c       user
    2.6244    2     0      0   100.000               pattaro
    2.8992    6     3      0     1.700                cygnus
    2.7501   33   206      3     0.160                bootes
    1.2872   11    69    209     0.158             hal__9000
    1.3019   20   127    279     0.157           themiconman
    2.1900    8    51    406     0.156         simpleelegant
    1.5413    2    12    458     0.156             javibear1
    2.7485   12    77    470     0.155                fornax
    1.5468    2    12    548     0.155              javibear
    1.3506    9    59    561     0.152                expdes
    1.2492  159  1046    620     0.152           steppenwolf
    2.6392    2    13   1666     0.151           blockrocket
    1.5396    2    13   1679     0.151               hornsby
    1.2634  200  1324   1693     0.151     joseph_schumpeter
    1.2535   15    99   3017     0.151                iam123
    1.2470  100   662   3116     0.151            quantverse
    3.0629   50   333   3779     0.150               kreator
    3.0395   50   333   4112     0.150                timboy
    3.0368   50   333   4445     0.150                ragnar
    1.2532   24   160   4779     0.150              daenris2
    1.2514   15   100   4939     0.150            nchamukong
    1.2585   50   335   5039     0.149              tomservo
    1.2478  100   689   5374     0.145                   bps
    1.2472    7    48   6064     0.145                ratnan
    1.2496   10    69   6112     0.144                srupal
    1.3077   10    70   6182     0.142       zbieram_na_piwo
    1.8337   50   354   6252     0.141       bookofillusions
    1.2474   60   425   6607     0.141          sebastian001
    1.2471   50   354   7032     0.141                 alisa
    2.4157    1     7   7387     0.140             thinker25
    1.8979   27   196   7394     0.140              assembly
    1.2541   50   357   7590     0.140                theafh
    1.2527   50   357   7947     0.140             karl_marx
    1.2512    7    50   8304     0.138                  avis
    1.2559  250  1838   8355     0.136                phorex
    2.1287    3    22  10193     0.134             enumerati
    1.2543    1     9  10216     0.134              tyler333
    2.1393   30   225  10226     0.133                   luz
    2.1370   30   225  10451     0.133                  luz2
    2.1364   30   225  10677     0.133                   mel
    2.1359   30   225  10902     0.133                  cla1
    1.2683    5    37  11128     0.133                tyler3
    1.6054    1     7  11166     0.132                  dnet
    1.8272   50   413  11173     0.121        onesimplenight
    1.7243    4    33  11586     0.121           acai_sorbet
    1.8979   30   257  11619     0.120              rymrgand
    1.7847    5    45  11877     0.111                  alai
    2.9983    1     9  11922     0.110         objectscience
    1.8979   23   209  11931     0.110                 sparq
    1.8979   26   236  12140     0.110                nmrnmr
    1.7787   22   200  12377     0.110          nasdaqjockey
    1.8433   40   396  12577     0.101                    dg
    1.7991    1    10  12973     0.100              algonell
    1.6750   45   500  12983     0.090    no_formal_training
    1.2780   40   487  13483     0.082             bulldozer
    2.1418    1    13  13970     0.076               elon_ai
    2.1294    4    52  13984     0.076            predictavo
    2.1437    1    16  14036     0.061             witnessai
    1.2906   80  1333  14053     0.060         forever_young
    1.4781   51  1000  15386     0.051  data_science_machine
    3.0934    5   100  16386     0.050                mmfine
    3.0918    5   100  16486     0.050               mmfine1
    3.0889    5   100  16586     0.050            washington
    1.5902    2    40  16686     0.050               expdes2
    2.5031    1    30  16726     0.033                  bor3
    2.5213    1    33  16756     0.030                  bor1
    2.5177    1    33  16790     0.030                  bor2
    2.1559   25  1000  16823     0.025       glasperlenspiel

You can optionally specify the round number and the column by which to sort.

You can add your own custom columns by grabbing the dataframe and inserting
whatever columns you like::

    >>> df = nx.get_stakes()
    >>> df['mycolumn'] = ...
