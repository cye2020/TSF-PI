# 주가 데이터, 환율

import FinanceDataReader as fdr

kospi = fdr.DataReader('KS11', '2000')[['Close']]
usd_krw = fdr.DataReader('USD/KRW', '2000')

print(kospi)
print(usd_krw)


