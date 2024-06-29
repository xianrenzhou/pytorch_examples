import pandas as pd
data = {
    'x': [],
    'y': []
}
for i in range(100):
    j = i**2 + i + 1
    data['x'].append(i)
    data['y'].append(j)
df = pd.DataFrame(data)
df.to_csv("./data/model_quadratic_function.csv",index=False)
