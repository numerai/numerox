Staking information
===================

Numerox provides information about the staking tournament::

    >>> import numerox as nx
    >>> nx.show_stakes(tournament='bernie', mark_user='steppenwolf')
                          days    s   soc   cumsum c      mark
    user
    jvk                   1.5177   0     0      0   0.700
    prosperity            0.5283   0     0      0   0.450
    unmarkedlearner       1.9310   0     0      0   0.400
    dustoff               1.9655   5    14      1   0.350
    cosmo                 0.2776   0     2     15   0.150
    onesimplenight        0.6861   1     9     17   0.110
    numbergamesai2        0.6383   0     1     27   0.110
    slyfox                1.9588   0     1     28   0.100
    andrei113             1.1434   0     1     29   0.100
    milanai               0.8345   0     1     30   0.100
    crysis                0.7821   1    10     31   0.100
    deltron               0.6383   0     5     41   0.100
    jackolantern          0.3637   0     2     46   0.100
    noone                 0.1450   5    50     48   0.100   new
    sjn                   0.6988   0     2     98   0.085
    dg                    0.6946   1    14    101   0.071
    hoi                   0.5262   1    22    115   0.050
    kreator               1.9097   2    48    137   0.041
    fearindex             0.7387   5   121    186   0.041
    parmenas              1.5905   1    40    308   0.040
    warg_1                0.2605   1    25    348   0.040
    lothlorien            0.2576   1    25    373   0.040
    wacax                 1.5634   0     6    398   0.035
    anna2                 0.7797   1    31    404   0.032
    anna1                 0.6673   2    62    435   0.032
    theorist13            0.3629   3    96    498   0.031
    smirmik3              0.1412   2    95    594   0.021   new
    mlt                   0.8278   0    25    690   0.020
    accountnumber3        0.1452   0    10    715   0.020   new
    mmfine                0.1447   2   105    725   0.020   new
    mmfine1               0.1444   2   105    830   0.020   new
    washington            0.1443   2   105    935   0.020   new
    glasperlenspiel       0.1536  14   736   1040   0.019   new
    smirmik2              0.1431   2   105   1776   0.019   new
    steppenwolf           0.1547   6   333   1882   0.018  <<<<
    data_science_machine  0.1665  14   823   2215   0.017
    smirmik               0.1556   2   117   3039   0.017
    monai                 0.1601   2   125   3156   0.016
    innmrwetrust          1.3307   3   200   3281   0.015
    themicon              0.1767   1    66   3481   0.015
    <snip>

You can optionally specify the round number, the column by which to sort,
whether to use integers for some columns, and whether to mark a user and flag
any stakes made after that of the marked user.

You can add your own custom columns by grabbing the dataframe and inserting
whatever columns you like::

    >>> df = nx.get_stakes()
    >>> df['mycolumn'] = ...
