# M1 통화량

import requests
import pandas as pd


def load_oecd_data(dataset: str, keys: list, param: list):
        base = 'http://stats.oecd.org/sdmx-json/data'
        
        servicekey = '.'.join(x[1] for x in keys)
        query = '&'.join('='.join(x) for x in param)

        url = f'{base}/{dataset}/{servicekey}/all?{query}'
        print(url)

        r: dict = requests.get(url).json()

        date_list = r['structure']['dimensions']['observation'][0]['values']
        dates = pd.to_datetime([x['id'] for x in date_list])

        title = r['structure']['dimensions']['series'][0]['values'][0]['name']

        df = pd.DataFrame()
        finder = ":".join(["0" for _ in range(len(keys))])
        s_list = r['dataSets'][0]['series'][finder]['observations']
        df[title] = pd.DataFrame([s_list[val][0] for val in sorted(s_list, key=int)])
        df.index = dates
        # print(title)
        print(df)


# M1 통화량 ================================================================================
dataset = 'MEI_FIN'

keys = [('subject', 'MANM'),
         ('country', 'KOR'),
         ('freq', 'M'), 
        ]


param = [('startTime' , '2001'),
         ('endTime', '2024'),
        ]
# ==========================================================================================


# 장기 국채 이자율 ==========================================================================
dataset = 'KEI'

keys = [('subject', 'IRLTLT01'),
         ('country', 'KOR'),
         ('measure', 'ST'),
         ('freq', 'M'), 
        ]


param = [('startTime' , '2001'),
         ('endTime', '2024'),
        ]
# ==========================================================================================



load_oecd_data(dataset, keys, param)