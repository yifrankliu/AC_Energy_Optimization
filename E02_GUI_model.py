import pandas as pd
import pickle
import numpy as np
import A02_optimization_model
from A03_room_temp_model import Room
from A02_optimization_model import Model_parameters



class AC_control_model:
    def __init__(self, data_temp_file, num_people_in_classroom_file, control_period_hr, model_para):

        self.model_para = model_para
        self.temp_data = pd.read_csv(data_temp_file)
        self.start_day = 4
        self.num_rooms = model_para.num_rooms
        self.Last_step_temp_to_used = model_para.num_horizon_intervals
        self.num_intervals_to_pred = model_para.num_horizon_intervals

        self.start_temp_indx = self.start_day * 24 * 60 // model_para.interval_length_min

        self.control_period = control_period_hr * 60 // 5

        with open(r"model/Temperature_predictor.pickle", "rb") as input_file:
            self.temp_pred = pickle.load(input_file)

        self.num_people_in_room = pd.read_excel(num_people_in_classroom_file)

        self.Rooms_obj = {}

        for r in range(self.num_rooms):
            self.Rooms_obj[r] = Room(R = model_para.R[r], C_air = model_para.C_air, M_air = model_para.M_air[r],
                                     start_temp = self.temp_data.iloc[self.start_temp_indx, 1], room_id = r)

        self.initial_temp_room = {}
        self.res = {'room': [], 'time': [], 'T_Out': [], 'Q_Ex': [], 'T_room': [], 'num_people_in_room': [],
               'power_usage': [], 'Q_AC': []}


    def run_model_step(self, t, Q_Ex=None, Q_AC=None):

        temp_model_x = self.temp_data.iloc[
                       self.start_temp_indx + t - self.Last_step_temp_to_used:self.start_temp_indx + t]
        T_Out = {}
        y = {}
        # pred tempature:msg=0
        for next in range(self.num_intervals_to_pred):
            model_name = f'next_{next * 5}_{(next + 1) * 5}_min'
            T_Out[next] = self.temp_pred[model_name].predict(np.array([temp_model_x.iloc[:, 1]]))[0]
            for r in range(self.num_rooms):
                try:
                    y[(next, r)] = self.num_people_in_room.loc[(self.num_people_in_room['Room'] == r) & (
                                self.num_people_in_room['Time_id'] == next + t), 'Num_people'].iloc[0]
                except:
                    y[(next, r)] = 0

        if t == 0:
            for r in range(self.num_rooms):
                self.initial_temp_room[r] = self.temp_data.iloc[self.start_temp_indx, 1]
                self.res['room'].append(r)
                self.res['time'].append(t)
                self.res['T_Out'].append(T_Out[0])
                self.res['num_people_in_room'].append(y[(0, r)])
                self.res['power_usage'].append(0)
                self.res['Q_AC'].append(0)
                self.res['T_room'].append(self.initial_temp_room[r])
                self.res['Q_Ex'].append(0)
        else:
            for r in range(self.num_rooms):
                Q_Ex_r = Q_Ex[(0, r)].varValue
                Q_AC_r = Q_AC[(0, r)].varValue
                if Q_AC_r >= 0:
                    Power_usage = Q_AC_r / self.model_para.heating_per_5min * self.model_para.power_usage_heating
                else:
                    Power_usage = (
                                      -Q_AC_r) / self.model_para.cooling_per_5min * self.model_para.power_usage_cooling

                self.Rooms_obj[r].temp_update(Q_Ex_r, Q_AC_r)
                self.initial_temp_room[r] = self.Rooms_obj[r].T

                self.res['room'].append(r)
                self.res['time'].append(t)
                self.res['T_Out'].append(T_Out[0])
                self.res['num_people_in_room'].append(y[(0, r)])
                self.res['power_usage'].append(Power_usage)
                self.res['Q_AC'].append(Q_AC_r)
                self.res['T_room'].append(self.initial_temp_room[r])
                self.res['Q_Ex'].append(Q_Ex_r)

        Q_Ex, Q_AC = A02_optimization_model.Optimization_control(T_Out, y, self.initial_temp_room, self.model_para)
        return Q_Ex, Q_AC, self.res


    def run_model(self, show_time_step=True):

        Q_Ex, Q_AC = 0, 0
        for t in range(self.control_period-1):
            if show_time_step:
                print('current time', t)
            Q_Ex, Q_AC, _ = self.run_model_step(t, Q_Ex, Q_AC)

        res_df = pd.DataFrame(self.res)
        res_df = res_df.sort_values(['room','time'])
        res_df.to_csv('results/optimal_solution.csv',index=False)

if __name__ == '__main__':
    num_people_in_classroom_file = "data/num_people_in_classroom.xlsx"
    data_temp_file = 'data/temperature_final_5_min.csv'
    model_para = Model_parameters()
    model = AC_control_model(data_temp_file=data_temp_file, num_people_in_classroom_file=num_people_in_classroom_file,
                             control_period_hr=24, model_para=model_para)
    model.run_model()