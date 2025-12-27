import pandas as pd
import matplotlib.pyplot as plt

save_fig = 1

temp_summer = pd.read_csv('results/optimal_solution.csv')
temp_winter = pd.read_csv('results/optimal_solution_winter.csv')

T_out_summer = temp_summer.loc[temp_summer['room'] == 0, 'T_Out'].values
T_out_winter = temp_winter.loc[temp_winter['room'] == 0, 'T_Out'].values

font_size = 18
fig, ax1 = plt.subplots(figsize=(15, 6))

time_id = temp_summer.loc[temp_summer['room'] == 0, 'time'].values
x = time_id


time_to_str = {0: '00:00', 48: '04:00', 96: '08: 00',
               144: '12:00', 192: '16:00', 240: '20:00', 288: '24:00'}
# time_to_str = {0: '00:00', 48: '04:00'}


temp = ax1.plot(x, T_out_summer, color='b', marker='s', lw=2, label='Summer')  # marker='s'
temp2 = ax1.plot(x, T_out_winter, color='r',  marker='o', lw=2, label='Winter')  # marker='s'

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
ax1.set_ylim([10, 35])


ax1.legend(loc='upper right', fontsize=font_size )
plt.tight_layout()
if save_fig == 1:
    plt.savefig(f'img/outside_temp.jpg', dpi=200)
else:
    plt.show()
