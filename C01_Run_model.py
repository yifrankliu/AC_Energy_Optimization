import pandas as pd
import pickle
import numpy as np
import A02_optimization_model
from A03_room_temp_model import Room
from A02_optimization_model import R, M_air, C_air


temp_data = pd.read_csv('data/temperature_final_5_min.csv')
start_day = 4

num_rooms = 2

Last_step_temp_to_used = 12
num_intervals_to_pred = 12

start_temp_indx = start_day * 24 * 60 // 5

control_period = 24 * 60 // 5

with open(r"model/Temperature_predictor.pickle", "rb") as input_file:
    temp_pred = pickle.load(input_file)

num_people_in_room = pd.read_excel("data/num_people_in_classroom.xlsx")

Rooms_obj = {}

for r in range(num_rooms):

    Rooms_obj[r] = Room(R = R[r], C_air = C_air, M_air = M_air[r], start_temp = temp_data.iloc[start_temp_indx, 1], room_id = r)


initial_temp_room = {}

res = {'room': [], 'time': [], 'T_Out': [], 'Q_Ex': [], 'T_room': [], 'num_people_in_room': [],
       'power_usage': [], 'Q_AC': []}

for t in range(control_period-1):
    print('current time', t)

    temp_model_x = temp_data.iloc[start_temp_indx+t - Last_step_temp_to_used:start_temp_indx+t]
    T_Out = {}
    y = {}
    # pred tempature:msg=0
    for next in range(num_intervals_to_pred):
        model_name = f'next_{next * 5}_{(next + 1) * 5}_min'
        T_Out[next] = temp_pred[model_name].predict(np.array([temp_model_x.iloc[:,1]]))[0]
        for r in range(num_rooms):
            try:
                y[(next, r)] = num_people_in_room.loc[(num_people_in_room['Room'] == r) & (num_people_in_room['Time_id'] == next+t), 'Num_people'].iloc[0]
            except:
                y[(next, r)] = 0

    if t == 0:
        for r in range(num_rooms):
            initial_temp_room[r] = temp_data.iloc[start_temp_indx, 1]
            res['room'].append(r)
            res['time'].append(t)
            res['T_Out'].append(T_Out[0])
            res['num_people_in_room'].append(y[(0,r)])
            res['power_usage'].append(0)
            res['Q_AC'].append(0)
            res['T_room'].append(initial_temp_room[r])
            res['Q_Ex'].append(0)
    else:
        for r in range(num_rooms):
            Q_Ex_r = Q_Ex[(0,r)].varValue
            Q_AC_r = Q_AC[(0,r)].varValue
            if Q_AC_r >= 0:
                Power_usage = Q_AC_r / A02_optimization_model.heating_per_5min * A02_optimization_model.power_usage_heating
            else:
                Power_usage = (-Q_AC_r) / A02_optimization_model.cooling_per_5min * A02_optimization_model.power_usage_cooling

            Rooms_obj[r].temp_update(Q_Ex_r, Q_AC_r)
            initial_temp_room[r] = Rooms_obj[r].T

            res['room'].append(r)
            res['time'].append(t)
            res['T_Out'].append(T_Out[0])
            res['num_people_in_room'].append(y[(0,r)])
            res['power_usage'].append(Power_usage)
            res['Q_AC'].append(Q_AC_r)
            res['T_room'].append(initial_temp_room[r])
            res['Q_Ex'].append(Q_Ex_r)

    Q_Ex, Q_AC = A02_optimization_model.Optimization_control(T_Out, y, initial_temp_room)



res_df = pd.DataFrame(res)
res_df = res_df.sort_values(['room','time'])

res_df.to_csv('results/optimal_solution.csv',index=False)
