
import numpy as np
import pandas as pd

arr = np.array([[12 ,15 ,18 ,19 ,20],
               [10, 16, 7, 18, 20],  
               [20, 12, 24, 11, 14]])

brr = np.array([[11 ,11 ,11 ,11 ,11],
               [22, 22, 22, 22, 22],  
               [33, 33, 33, 33, 33]])

x = np.mean(arr, axis=0, dtype=np.float32)
print("Output array: ", x)

ts = pd.read_csv(f"testDF-0.csv")

tt = pd.read_csv(f"testDF-1.csv")

ttdf = pd.concat([tt, ts]).reset_index(drop=True)
ttdf = ttdf.groupby(["image", "target"]).distances.max().reset_index()
ttdf = ttdf.sort_values("distances", ascending=False).reset_index(drop=True)

val = 42
