#%%
from pytrends.request import TrendReq
import matplotlib.pyplot as plt

pytrends = TrendReq(hl='ko-KR', tz=540)
kw_list = ['samsung']
pytrends.build_payload(kw_list, timeframe='2022-01-01 2023-09-03', geo='KR')
df = pytrends.interest_over_time()
df.plot()
print(df)

# %%
