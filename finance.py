import FinanceDataReader as fdr

kospi = fdr.DataReader('KS11')
usd_krw = fdr.DataReader('USD/KRW', '1995')

print(kospi)
print(usd_krw)


