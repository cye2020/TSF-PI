# GDP

import wbdata
import datetime

# Set the date range
start_date = datetime.datetime(2000, 1, 1)
end_date = datetime.datetime(2020, 1, 1)

# Fetch GDP data (current US$) for all countries
data = wbdata.get_data("NY.GDP.MKTP.CD", country=('KOR'), data_date=(start_date, end_date))

for entry in data:
    print(entry['country']['id'], entry['country']['value'], entry['date'], entry['value'])