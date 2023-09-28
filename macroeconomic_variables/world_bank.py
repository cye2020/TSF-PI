'''
Import macroeconomic variables data from World bank
Unit: USD Billion
'''

import wbdata
from datetime import datetime
import pandas as pd





    
def get_data(dataset: str, start_date: datetime, end_date: datetime, name: str = 'Value'):
    wb = wbdata.get_data(dataset, country=('KOR'), data_date=(start_date, end_date))

    data_list = []
    
    for entry in wb:
        df = pd.DataFrame({'Date': [datetime.strptime(entry['date'], '%Y')], name: [round(entry['value']/1000000000,2)]})
        data_list.append(df)
    
    data = pd.concat(data_list, ignore_index=True)
    data.set_index('Date', inplace=True)
    return data


if __name__=='__main__':
    dataset = "NY.GDP.MKTP.CD"
    # Set the date range
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    name = 'GDP'
    
    data = get_data(dataset, start_date, end_date, name)
    
    print(data)