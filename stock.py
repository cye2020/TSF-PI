#%% 
from tqdm import tqdm
import requests
import time
import pandas as pd
from pathlib import Path
import FinanceDataReader as fdr
from pykrx.website import krx
from pykrx import stock
import mplfinance as mpf
import sys


df = fdr.DataReader('KS11', '2015')
print(df)
# %%
