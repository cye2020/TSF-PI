import pandas as pd
import numpy as np
from pandas import date_range


def fill_missing_dates(data: pd.DataFrame):
    """
    Fills missing dates in a dataframe with the last valid date.

    Args:
        data (pd.DataFrame): Dataframe with missing dates.

    Returns:
        pd.DataFrame: Dataframe with filled missing dates.
    """
    missing_dates = get_missing_dates(data)
    missing_df = pd.DataFrame({
        'Date': missing_dates
    })
    
    for col in data.columns[1:]:
        missing_df[col] = np.nan
    data = pd.concat([data, missing_df], ignore_index=True).sort_values('Date').reset_index(drop=True)
    
    return missing_dates, data


def get_missing_dates(data: pd.DataFrame):
    start_date = data.iloc[0]['Date']
    end_date = data.iloc[-1]['Date']
    date_range_all = date_range(start_date, end_date).strftime('%Y-%m-%d')
    data_dates = set(data['Date'])
    missing_dates = list(date_range_all.difference(data_dates))
    
    return missing_dates


if __name__=='__main__':
    data = pd.read_csv('./data/kospi.csv').dropna()
    df = fill_missing_dates(data)
    print(df[:10])