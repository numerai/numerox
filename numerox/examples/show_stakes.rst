Staking information
===================

Numerox provides information about the staking tournament::

    >>> import numerox as nx
    >>> nx.show_stakes(mark_user='bps')
                          days    s    soc   cumsum  c     mark
    user
    gopovo                2.2766    3     5      0   0.730
    npc                   2.2807    1     3      5   0.300
    shimco                3.1578    1     3      8   0.260
    hb_scout              2.9623    2     7     12   0.256
    hb_exp                2.9640    2     7     20   0.255
    timboy                3.1049   25   131     27   0.190
    costello              2.2760    1     7    159   0.136
    hb                    2.9630    2    14    166   0.134
    yty                   2.2812   50   381    181   0.131
    zuz                   2.2781   50   381    563   0.131
    no_formal_training    2.9833   13   100    945   0.130
    themicon              2.3197    5    40   1045   0.125
    themiconman           2.3193    5    40   1085   0.124
    themiconmanweb        2.3187    5    40   1125   0.123
    chrissly31415         3.0337   10    95   1166   0.105
    wsw                   2.2748   50   490   1261   0.102
    justaddmorelayers     2.2766  100   990   1751   0.101
    witnessai             4.1025    2    20   2741   0.100
    dnum1_2               3.4168    1    10   2761   0.100
    kreator               3.1093   20   200   2771   0.100
    ragnar                3.1028   20   200   2971   0.100
    abriosi               2.6279    9    90   3171   0.100
    tyler3                3.9674    2    22   3261   0.090
    niccholas             2.3035   18   210   3283   0.088
    anna2                 2.3053    2    22   3493   0.087
    anna1                 2.3052    2    22   3516   0.087
    quantverse            2.2745  100  1149   3539   0.087   new
    data_science_machine  2.2746   50   581   4689   0.086   new
    theafh                2.3296   10   117   5270   0.085
    karl_marx             2.3280   10   117   5388   0.085
    joseph_schumpeter     2.3268   10   117   5505   0.085
    milgram               2.2822    3    46   5623   0.085
    narf                  2.2744  101  1202   5670   0.084   new
    bps                   2.2748  100  1204   6872   0.083  <<<<
    steppenwolf           2.2748   50   602   8077   0.083
    alisa                 2.2745   50   602   8680   0.083   new
    euageney              2.2753   23   282   9282   0.082
    zbieram_na_piwo       2.2745    1    12   9565   0.082   new
    <snip>

You can optionally specify the round number, the column by which to sort,
whether to use integers for some columns, and whether to mark a user and flag
any stakes made after that of the marked user.

You can add your own custom columns by grabbing the dataframe and inserting
whatever columns you like::

    >>> df = nx.get_stakes()
    >>> df['mycolumn'] = ...
