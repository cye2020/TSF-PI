import requests
import pandas as pd
from typing import Literal
from datetime import datetime


class OecdQuery:
    def __init__(self, dataset: Literal) -> None:
        self.dataset = dataset
        self.keys = []
        self.param = []
    
    def set_keys(self, subject: Literal = '', country: Literal = '', measure: Literal = '', freq: Literal = ''):
        params = ['subject', 'country', 'measure', 'freq']
        values = [subject, country, measure, freq]
        self.keys = [(param, value) for param, value in zip(params, values) if value]
    
    def set_param(self, start_date: datetime, end_date: datetime):
        self.param = [
            ('startTime' , start_date.strftime('%Y-%m-%d')),
            ('endTime', end_date.strftime('%Y-%m-%d')),
        ]
        

def get_data(query: OecdQuery, name: Literal = 'Value'):
    base = 'http://stats.oecd.org/sdmx-json/data'
    
    servicekey = '.'.join(x[1] for x in query.keys)
    parameters = '&'.join('='.join(x) for x in query.param)

    url = f'{base}/{query.dataset}/{servicekey}/all?{parameters}'
    r: dict = requests.get(url).json()

    date_list = r['structure']['dimensions']['observation'][0]['values']
    dates = pd.to_datetime([x['id'] for x in date_list])
    # title = r['structure']['dimensions']['series'][0]['values'][0]['name']

    data = pd.DataFrame()
    finder = ":".join(["0" for _ in range(len(query.keys))])
    s_list = r['dataSets'][0]['series'][finder]['observations']
    data[name] = pd.DataFrame([s_list[val][0] for val in sorted(s_list, key=int)])
    data['Date'] = dates
    data = data[['Date', name]]
    data.set_index('Date', inplace=True)
    return data


if __name__ == '__main__':
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2024, 1, 1)
    LIR_query = OecdQuery('KEI')
    LIR_query.set_keys(subject='IRLTLT01', country='KOR', measure='ST', freq='M')
    LIR_query.set_param(start_date, end_date)
    
    M1_query = OecdQuery('MEI_FIN')
    M1_query.set_keys(subject='MANM', country='KOR', freq='M')
    M1_query.set_param(start_date, end_date)

    LIR_data = get_data(LIR_query, 'LIR')
    print(LIR_data)
    
    M1_data = get_data(M1_query, 'M1')
    print(M1_data)