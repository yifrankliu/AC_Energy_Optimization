import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

data = pd.read_csv('data/temperature_final.csv')
data = data.groupby(['Date time'])['Air Temperature(degree Celsius)'].mean().reset_index()
data['time_str'] = data['Date time'].astype('str')
data['temperature'] = data['Air Temperature(degree Celsius)'] * 1.15
data['min'] = data['time_str'].apply(lambda x: x[-2:])
data['hour'] = data['time_str'].apply(lambda x: x[-4:-2])
data['day'] = data['time_str'].apply(lambda x: x[-6:-4])
data['min_num'] = data['min'].astype('int') + data['hour'].astype('int')*60 + data['day'].astype('int')*24*60
data['time_id'] = data['min_num'] // 5
data['time_id'] -= min(data['time_id'])

all_time_id = pd.DataFrame({'time_id':list(range(max(data['time_id']) + 1))})
final_data = all_time_id.merge(data[['time_id', 'temperature']], how = 'left')
final_data['temperature'] = final_data['temperature'].fillna(method='ffill')

new_temp = moving_avg(final_data['temperature'].values, 20)
# new_temp = np.hstack([new_temp, np.array([new_temp[-1]])])


final_data = final_data.head(len(new_temp))



final_data['temperature'] = new_temp

# plt.plot(final_data['time_id'],final_data['temperature'], 'r')
# plt.plot(final_data['time_id'],final_data['temp_mov_avg'], 'b')
# plt.show()

final_data.to_csv('data/temperature_final_5_min.csv',index=False)
# a=1

