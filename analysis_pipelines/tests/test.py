import pandas as pd
import numpy as np

df_main = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6],
                  "c": [7, 8, 9], "d": [10, 11, 12]})

df_0 = pd.Series({"a": 1, "b": 4})

# append two rows to df_main
df_main = df_main.append(df_0, ignore_index=True)

