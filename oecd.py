import requests
import pandas as pd

base = 'http://stats.oecd.org/sdmx-json/data'

dataset = 'MEI_FIN'

keys = [('subject', 'MANM'),
         ('country', 'KOR'),
         ('freq', 'M'), 
        ]


param = [('startTime' , '2020'),
         ('endTime', '2023'),
        ]

servicekey = '.'.join(x[1] for x in keys)
query = '&'.join('='.join(x) for x in param)

url = f'{base}/{dataset}/{servicekey}/all?{query}'
print(url)

r: dict = requests.get(url).json()

date_list = r['structure']['dimensions']['observation'][0]['values']
dates = pd.to_datetime([x['id'] for x in date_list])

title = r['structure']['dimensions']['series'][0]['values'][0]['name']

df = pd.DataFrame()
s_list = r['dataSets'][0]['series']['0:0:0']['observations']
df[title] = pd.DataFrame([s_list[val][0] for val in sorted(s_list, key=int)])
df.index = dates
# print(title)
print(df)