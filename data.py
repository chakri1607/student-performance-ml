import pandas as pd

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Sleep": [6,6,7,7,8,8,9,9],
    "Marks": [35,40,50,60,65,70,75,80],
    "Pass":  [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)
print(df)

