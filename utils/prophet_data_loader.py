from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd



class ProphetDataLoader:
    def __init__(self):
        self.original = None  # Original data
        self._scaler = None  # Scaling Inst
        self.scaler = None
    
    def load_csv(self, path: str, end_date: str):
        data = pd.read_csv(path)
        data = data[data['Date'] <= end_date]
        data['Change'] = data['Kospi'].pct_change() * 100
        data.loc[0, 'Change'] = 0
        
        self.original = data.copy()
        return data
    
    def split_data(self, data: pd.DataFrame, split_date: str):
        # train, test 데이터 나누기
        train_data = data.loc[data['Date'] <= split_date].copy()
        test_data = data.loc[data['Date'] > split_date].copy()
        
        data.rename(columns={'Date': 'ds', 'Change': 'y'}, inplace=True)
        data['ds'] = pd.to_datetime(data['ds'])

        train_data.rename(columns={'Date': 'ds', 'Change': 'y'}, inplace=True)
        train_data['ds'] = pd.to_datetime(train_data['ds'])

        test_data.rename(columns={'Date': 'ds', 'Change': 'y'}, inplace=True)
        test_data['ds'] = pd.to_datetime(test_data['ds'])
    
        return train_data, test_data

    def standardScale(self, column_name: str):
        if self.scaler is not StandardScaler:
            self._scaler = StandardScaler()
            self.scaler = StandardScaler()
            self._scaler.fit(self.original)
            self.scaler.fit(self.original[[column_name]])
        data = self._scaler.transform(self.original[[column_name]])
        return data
    
    def minmaxScale(self, column_name: str):
        if self.scaler is not MinMaxScaler:
            self._scaler = MinMaxScaler()
            self.scaler = MinMaxScaler()
            self._scaler.fit(self.original)
            self.scaler.fit(self.original[[column_name]])
        data = self._scaler.transform(self.original[[column_name]])
        return data
    
    def inverseScale(self, data: pd.DataFrame, column_name: str):
        inverse_data = self.scaler.inverse_transform(data[column_name].values)
        return inverse_data

