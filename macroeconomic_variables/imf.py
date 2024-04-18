import requests
import pandas as pd

def get_data(indicators: dict = {}, countries = [], regions = [], groups = [], periods = []):
    url = 'https://www.imf.org/external/datamapper/api/v1/'
    indicators_path = '/'.join(indicators.keys())
    countries_path = '/'.join(countries)
    regions_path = '/'.join(regions)
    groups_path = '/'.join(groups)
    url = url + indicators_path + '/' + countries_path + '/' + regions_path + '/' + groups_path
    
    periods_query = {'periods': ','.join(periods)}
    r = requests.get(url, params=periods_query)
    results = r.json()['values']
    
    
    data = {'Date': []}
    data.update({k: [] for k in indicators})
    data = {k: [] for k in (countries + regions + groups)}
    
    for indicator, v in results.items():
        for k, result in v.items():
            for period, value in result.items():
                data[k].append(pd.DataFrame({'Date': [period], indicators[indicator]: [value]}))
            data[k] = pd.concat(data[k], ignore_index=True).sort_values('Date')
            data[k]['Date'] = pd.to_datetime(data[k]['Date'], format='%Y')
            data[k].set_index('Date', inplace=True)

    return data


if __name__ == '__main__':
    indicators = {'NGDP_RPCH': 'GDP'}
    countries = ['KOR']
    periods = ['2024']
    data = get_data(indicators, countries, periods=periods)['KOR']
    data['GDP'] = data['GDP'].apply(lambda x: round(x / 4, 2))
    print(data)