# 필요한 라이브러리와 모듈을 가져옴
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
from macroeconomic_variables import oecd, kosis, OecdQuery, imf


# 주 실행 스크립트 시작점
if __name__ == '__main__':
    # 데이터를 가져올 시작 날짜와 종료 날짜 설정
    start_date = datetime(2010, 1, 1)
    end_date = datetime.today()
    
    # KOSIS API 인증키를 읽어옴
    keys = pd.read_csv('/home/yeeun/skku/Graduate/TSF-PI/Key.csv', index_col=0)
    service_key = keys.loc['Kosis', 'Key']
    
    # KOSIS에서 분기 GDP 데이터 가져오기
    orgId = "301"
    tblId = 'DT_200Y002'
    freq = 'Q'
    gdp_data = kosis.get_data(service_key, orgId, tblId, freq, start_date, end_date, 'GDP')
    
    
    # IMF에서 연간 GDP 예상 데이터 가져오기
    indicators = {'NGDP_RPCH': 'GDP'}
    countries = ['KOR']
    periods = [str(end_date.year)]
    gdp_predict = imf.get_data(indicators=indicators, countries=countries, periods=periods)['KOR']
    gdp_predict['GDP'] = gdp_predict['GDP'].apply(lambda x: round(x / 4, 2)) # 연도 -> 분기로
    
    # GDP 예상 처음 날짜 조정 (이미 1분기가 지난 경우를 위해)
    predict_start_date = max(gdp_predict.index[0], gdp_data.index[-1] + timedelta(days=1))
    idx = list(gdp_predict.index)
    idx[0] = predict_start_date
    gdp_predict.index = idx
    
    # GDP 데이터 합치기
    gdp_data = pd.concat([gdp_data, gdp_predict])
    
    # KOSIS에서 전산업생산지수 데이터 가져오기
    
    # KOSIS API의 기타 설정
    orgId = "101"
    tblId = 'DT_1JH20201'
    freq = 'M'
    iaip_data = kosis.get_data(service_key, orgId, tblId, freq, start_date, end_date, 'IAIP')
    
    # OECD에서 장기 국채 이자율 데이터 가져오기
    lir_query = OecdQuery('KEI')
    lir_query.set_keys(subject='IRLTLT01', country='KOR', measure='ST', freq='M')
    lir_query.set_param(start_date, end_date)
    lir_data = oecd.get_data(lir_query, 'LIR')
    
    # OECD에서 M1 통화량 데이터 가져오기
    m1_query = OecdQuery('MEI_FIN')
    m1_query.set_keys(subject='MANM', country='KOR', freq='M')
    m1_query.set_param(start_date, end_date)
    m1_data = oecd.get_data(m1_query, 'M1')
    
    # FinanceDataReader를 사용하여 코스피 지수 데이터 가져오기
    kospi_data = fdr.DataReader('KS11', start_date, end_date)[['Close']].dropna()
    kospi_data.columns = ['KOSPI']
    
    # FinanceDataReader를 사용하여 USD/KRW 환율 데이터 가져오기
    usd_krw_data = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']].dropna()
    usd_krw_data.columns = ['USD/KRW']
    
    # 모든 경제 관련 데이터를 하나의 리스트에 저장
    macroeconimic_data = [usd_krw_data, gdp_data, iaip_data, lir_data, m1_data]
    
    # 데이터 병합을 위해 kospi_data의 사본을 만듦
    data = kospi_data.copy()

    # kospi_data의 각 날짜에 대해 나머지 경제 데이터프레임들의 최근 값을 병합
    for df in macroeconimic_data:
        data = pd.merge_asof(
            data.sort_index(), df.sort_index(),
            left_index=True, right_index=True,
            direction='backward'
        )

    # Disease: COVID-19 변수 추가
    data['Disease'] = 0  # 모든 행에 대해 초기값을 0으로 설정

    # 메르스 유행 기간은 1로 설정
    data.loc[('2015-12-23' >= data.index) & (data.index >= '2015-05-20'), 'Disease'] = 1
    
    # COVID-19기간은 1로 설정
    data.loc[('2022-08-31' >= data.index) & (data.index >= '2020-01-20'), 'Disease'] = 1
    

    # 최종 병합된 데이터 저장
    data.to_csv('./data/kospi.csv')

    monthly_data = data.resample('MS').first()
    monthly_data.to_csv('./data/monthly_kospi.csv')

