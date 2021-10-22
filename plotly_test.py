import pandas as pd
import plotly.express as px

df = pd.DataFrame(dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9], d=[1, 2, 3]))
fig = px.scatter_3d(df, x='a', y='b', z='c',
              color='d')
fig.show()