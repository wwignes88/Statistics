import numpy as np
import pandas as pd


arrays = [
    np.array(["m=2", "m=2", "m=2", "m=2", \
              "m=3", "m=3", "m=3", "m=3", \
              "m=4", "m=4", "m=4", "m=4", \
              "m=6", "m=6", "m=6", "m=6", \
              "m=5", "m=5", "m=5", "m=5", \
              "m=7", "m=7", "m=7", "m=7", \
              "m=8", "m=8", "m=8", "m=8", \
              "m=9", "m=9", "m=9", "m=9"]),
    np.array(["0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10",\
              "0.01", "0.025", "0.05", "0.10"]),
]


index=["n=9", "n=10", "n=11", "n=12", "n=13", "n=14"]
MWM    = pd.DataFrame(np.random.randn(32, 6),index=arrays, columns=index)
#print(f'MWM table: \n{MWM}')

def MWM_stat(n,m,α):
    n = np.max([n,m]) ; m = np.min([n,m])
    val = MWM[f"n={n}"][f"m={m}"][f"{.01}"]
    return val

#n,m,α = 9,2,.01
#MWM_stat(n,m,α )


#================ Wilcoxon signed rank test ===========

Warrays = np.array(["n=4", "n=5",   "n=6",   "n=7", \
                   "n=8", "n=9",   "n=10", "n=11", \
                   "n=12", "n=13", "n=14", "n=15", \
                   "n=16", "n=17", "n=18", "n=19", \
                "n=20"])
    
d = np.matrix('None None None None 0  2  4;\
               None None None  0   2  3  7;\
               None None  0    2   3  5  10;\
               None 0     2    3   5  8  13;\
               0  1      3    5   8  11 17;\
               1  3      5    8   10 14 22;\
               3  5      8    10  14 18 27;\
               5  7      10   13  17 22 32;\
               7  9      13   17  21 27 38;\
               9  12     17   21  26 32 45;\
               12 15     21   25  31 38 52;\
               15 19     25   30  36 44 59;\
               19 23     29   35  42 50 67;\
               23 27     34   41  48 57 76;\
               27 32     40   47  55 65 85;\
               32 37     46   53  62 73 94;\
               37 43     52   60  69 81 104') 



Windex=["0.005", "0.01", "0.025", "0.05", "0.10", "0.20", "0.50"]
Wilcoxon    = pd.DataFrame(d, index=Warrays, columns=Windex)
Wilcoxon = Wilcoxon.rename_axis( columns="α")

def Wilcoxon_stat(n,α):
    val = Wilcoxon[f"{α}"][f"n={n}"]
    return val

#n,α = 9,.01
#Wilcoxon_stat(n,α)

# find p val of T = sum of signed ranks
def Wilcoxon_loc(T):
    i = 0
    while i < len(Windex):
        α_val = Windex[i]
        col   = Wilcoxon[f'{α_val}'].values.tolist()
        j = 0
        while j < len(Warrays):
            if col[j] == T:
                return α_val
            j += 1
        i += 1
#n = Wilcoxon_loc(81)
#print(f'\n n = \n{n}')



#===========================================













