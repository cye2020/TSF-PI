from PublicDataReader import Kosis
import pandas as pd
from typing import Literal
from datetime import datetime, timedelta

def get_data(
        service_key: str, 
        orgId: str, tblId: str, 
        freq: str, start_date: datetime, end_date: datetime,
        name: str = 'Value'
    ):
    # 인스턴스 생성하기
    api = Kosis(service_key)
    
    start_date = start_date - timedelta(days=1) 
    df: pd.DataFrame = api.get_data(
        "통계자료",
        orgId = orgId,
        tblId = tblId,
        itmId = "ALL",
        objL1 = "ALL",
        prdSe = freq,
        startPrdDe=start_date.strftime('%Y%m%d'),
        endPrdDe=end_date.strftime('%Y%m%d'),
    )
    
    data = df[['수록시점', '수치값']]
    data.columns = ['Date', name]
    
    if freq == 'Q':
        data.loc[:, 'Date'] = data['Date'].apply(lambda x: x[:-2] + str(3 * int(x[-1]) - 2))
    data.loc[:, 'Date'] = pd.to_datetime(data['Date'], format='%Y%m')
    data = data.drop_duplicates(subset=['Date'])
    data.set_index('Date', inplace=True)
    return data


if __name__=='__main__':
    # KOSIS 공유서비스 Open API 사용자 인증키
    keys= pd.read_csv('/home/yeeun/skku/Graduate/TSF-PI/Key.csv', index_col=0)
    service_key = keys.loc['Kosis', 'Key']
    
    orgId = "101"
    tblId = 'DT_1JH20201'
    freq = 'M'
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    data = get_data(service_key, orgId, tblId, freq, start_date, end_date, '전산업지수')
    print(data)