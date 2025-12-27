import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

case = 'summer' # summer
if case == 'summer':
    data_temp_file = 'data/temperature_final_5_min.csv'
    with open(r"model/Temperature_predictor.pickle", "rb") as input_file:
        temp_pred = pickle.load(input_file)
elif case == 'winter':
    data_temp_file = 'data/temperature_final_5_min_winter.csv'
    with open(r"model/Temperature_predictor_winter.pickle", "rb") as input_file:
        temp_pred = pickle.load(input_file)
else:
    print('case not define')
    exit()
interval_length_min = 5
start_day = 4
control_period = 24 * 60 // interval_length_min
temp_data = pd.read_csv(data_temp_file)
num_horizon_intervals = 12
num_intervals_to_pred = num_horizon_intervals
Last_step_temp_to_used = num_horizon_intervals
start_temp_indx = start_day * 24 * 60 // interval_length_min

num_rooms = 2
num_people_in_room = pd.read_excel("data/num_people_in_classroom.xlsx")


future_period = {}
x_id = []
actual_t = []
for t in range(control_period - 1):
    x_id.append(t)
    actual_t.append(temp_data['temperature'].iloc[start_temp_indx + t])
    temp_model_x = temp_data.iloc[start_temp_indx + t - Last_step_temp_to_used:start_temp_indx + t]
    T_Out = {}
    # pred tempature:msg=0
    for next_ in range(num_intervals_to_pred):
        model_name = f'next_{next_ * 5}_{(next_ + 1) * 5}_min'
        T_Out[next_] = temp_pred[model_name].predict(np.array([temp_model_x.iloc[:, 1]]))[0]
        if next_ not in future_period:
            future_period[next_] = {'x_id':[], 'temp_pred':[]}
        if t+next_+1 <= control_period-1-1:
            future_period[next_]['temp_pred'].append(T_Out[next_])
            future_period[next_]['x_id'].append(t+next_+1)
### plot

save_fig = 1

font_size = 18
fig, ax1 = plt.subplots(figsize=(15, 6))

time_id = list(range(control_period))
x = x_id


time_to_str = {0: '00:00', 48: '04:00', 96: '08: 00',
               144: '12:00', 192: '16:00', 240: '20:00', 288: '24:00'}
# time_to_str = {0: '00:00', 48: '04:00'}
if case == 'summer':
    temp = ax1.plot(x, actual_t, color='b', marker='s', lw=2, label=f'Actual')  # marker='s'
else:
    temp = ax1.plot(x, actual_t, color='r', marker='s', lw=2, label=f'Actual')  # marker='s'
# colors = sns.color_palette("rocket")
tranp = np.linspace(0.1, 0.9, len(future_period.keys()))
for next_ in future_period:
    temp = ax1.plot(future_period[next_]['x_id'], future_period[next_]['temp_pred'], color='g', alpha = 1-tranp[next_] , lw=2, label=f'Future {next_+1}')


# next_ = 11
# temp = ax1.plot(future_period[next_]['x_id'], future_period[next_]['temp_pred'], color='g', alpha = 1-tranp[next_] , lw=2, label=f'Future {next_+1}')  # marker='s'

x_ticks_time = [key for key in time_to_str]
x_ticks_str = [value for key, value in time_to_str.items()]
plt.xticks(x_ticks_time, x_ticks_str)

# ax1.plot([-1,3],[y1[0],y1[0]],'g--')
# ax2.plot([-1,3],[y2[0],y2[0]],'b--')

ax1.set_xlabel('Time of day', fontsize=font_size)
ax1.set_ylabel('Outside temperature', color='k', fontsize=font_size)

ax1.tick_params(labelsize=font_size)
# ax2.tick_params(labelsize=font_size)
plt.xticks(fontsize=font_size)

plt.xlim([-1, 289])
if case == 'summer':
    ax1.set_ylim([26, 37])
else:
    ax1.set_ylim([13, 25])


ax1.legend(loc='upper left', fontsize=font_size, ncol = 5)
plt.tight_layout()
if save_fig == 1:
    plt.savefig(f'img/temp_pred_{case}.jpg', dpi=200)
else:
    plt.show()


