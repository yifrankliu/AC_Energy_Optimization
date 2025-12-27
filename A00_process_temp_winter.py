import pandas as pd
import os
station_name = 'The Peak'

def get_time_id(time):
    time_id = str(time)
    num_zero = 4 - len(time_id)
    return '0'*num_zero + time_id

all_data = []

# summer
# date_range = ['20230101', '20230102', '20230103', '20230104', '20230105', '20230106', '20230107']


# winter:
date_range = ['20230102', '20230102', '20230103', '20230104', '20230105', '20230106', '20230107',  '20230108']


max_time = 2350
for date in date_range:
    for time in range(max_time):
        time_id = get_time_id(time)
        file_name ='data/' + date + '-' + time_id + '-' + 'latest_since_midnight_maxmin.csv'
        if os.path.exists(file_name):
            data = pd.read_csv(file_name)
            data_used = data.loc[data['Automatic Weather Station'] == station_name]
            try:
                data_used['Minimum Air Temperature Since Midnight(degree Celsius)'] = data_used['Minimum Air Temperature Since Midnight(degree Celsius)'].astype('float')
                data_used['Maximum Air Temperature Since Midnight(degree Celsius)'] = data_used['Maximum Air Temperature Since Midnight(degree Celsius)'].astype('float')
            except:
                data_used['Minimum Air Temperature Since Midnight(degree Celsius)'] = data_used['Minimum Air Temperature Since Midnight(degree Celsius)'].apply(lambda x: x.replace('*',''))
                data_used['Minimum Air Temperature Since Midnight(degree Celsius)'] = data_used[
                    'Minimum Air Temperature Since Midnight(degree Celsius)'].astype('float')
                a=1
            data_used['Air Temperature(degree Celsius)'] = (data_used['Maximum Air Temperature Since Midnight(degree Celsius)'] + data_used['Minimum Air Temperature Since Midnight(degree Celsius)'] ) / 2
            all_data.append(data_used)
        else:
            print(file_name, "not exist")


data_final = pd.concat(all_data)
data_final['Date time'] = data_final['Date time'].astype('int')
data_final = data_final.sort_values(['Date time'])
data_final.to_csv('data/temperature_final_winter.csv',index=False)



