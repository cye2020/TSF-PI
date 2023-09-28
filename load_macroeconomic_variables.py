from PublicDataReader import Kosis
import pandas as pd
from datetime import datetime

import requests
import FinanceDataReader as fdr


from macroeconomic_variables import oecd, kosis, OecdQuery
from macroeconomic_variables import world_bank as wb




if __name__ == '__main__':
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2024, 1, 1)
    
    # GDP ====================================================================================
    dataset = "NY.GDP.MKTP.CD"
    name = 'GDP'
    gdp_data = wb.get_data(dataset, start_date, end_date, name)
    # =========================================================================================
    
    
    # IAIP ====================================================================================
    # KOSIS 공유서비스 Open API 사용자 인증키
    keys= pd.read_csv('/home/yeeun/skku/Graduate/TSF-PI/Key.csv', index_col=0)
    service_key = keys.loc['Kosis', 'Key']
    
    orgId = "101"
    tblId = 'DT_1JH20201'
    freq = 'M'
    
    iaip_data = kosis.get_data(service_key, orgId, tblId, freq, start_date, end_date, 'IAIP')
    # =========================================================================================
    
    
    # Money Supply M1 =========================================================================
    
    lir_query = OecdQuery('KEI')
    lir_query.set_keys(subject='IRLTLT01', country='KOR', measure='ST', freq='M')
    lir_query.set_param(start_date, end_date)
    lir_data = oecd.get_data(lir_query, 'LIR')
    
    # =========================================================================================
    
    
    
    # Money Supply M1 =========================================================================
    
    m1_query = OecdQuery('MEI_FIN')
    m1_query.set_keys(subject='MANM', country='KOR', freq='M')
    m1_query.set_param(start_date, end_date)
    
    m1_data = oecd.get_data(m1_query, 'M1')
    
    # =========================================================================================
    
    
    # kospi index
    kospi_data = fdr.DataReader('KS11', start_date, end_date)[['Close']]
    kospi_data.columns = ['Kospi']
    
    # usd/krw currency
    usd_krw_data = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']]
    usd_krw_data.columns = ['USD/KRW']
    
    
    data_list = [kospi_data, gdp_data, iaip_data, lir_data, m1_data, usd_krw_data]
    result = kospi_data.copy()

    for df in data_list[1:]:
        result = pd.merge_asof(result.sort_index(), df.sort_index(),
                            left_index=True, right_index=True,
                            direction='backward')

    print(result)

    