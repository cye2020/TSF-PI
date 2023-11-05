from prophet import Prophet
import pandas as pd

def main():
    df = pd.read_csv('your_filepath.csv')
    df.rename(columns={'Date': 'ds', 'Kospi': 'y'}, inplace=True)

    prophet_model = Prophet(
        changepoint_prior_scale=0.01,
        yearly_seasonality=True,
        weekly_seasonality=True,
        holidays='US'
    )

    # 다른 변수들을 regressor로 추가
    additional_columns = ['USD/KRW', 'GDP', 'IAIP', 'LIR', 'M1', 'disease']
    for column in additional_columns:
        prophet_model.add_regressor(column)

    prophet_model.fit(df)

    future = prophet_model.make_future_dataframe(periods=365)
    # 예측하려는 기간에 대한 추가 변수의 값을 future 데이터프레임에 추가
    future = future.merge(df[['ds'] + additional_columns], on='ds', how='left')

    forecast = prophet_model.predict(future)
    fig = prophet_model.plot(forecast)

if __name__ == '__main__':
    main()