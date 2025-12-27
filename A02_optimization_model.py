from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable
import pandas as pd

# solve the control with 5 min intervals, for next 1 hour
interval_length = 5 * 60
num_rooms = 2
num_time_intervals = 12

GAMMA = 1e-3 # 0.01 # 0.00001


heating_per_5min = 907500  # J
cooling_per_5min = 825000   # J

power_usage_heating = 330000
power_usage_cooling = 300000 # J

#
C_air = 1000
M_air = {}
R = {}

global_target_temp = 24

Target_temp = {}
for r in range(num_rooms):
    Target_temp[r] = global_target_temp

M_air[0] = 312.4
R[0] = 0.036

M_air[1] = 165.4
R[1] = 0.031

def generate_dummy_inputs():

    T_Out = {}
    y = {}  # num people in class room
    for t in range(num_time_intervals):
        T_Out[t] = 25 + 0.1 * t

    for r in range(num_rooms):
        for t in range(num_time_intervals):
            y[(t,r)] = 10


    for r in range(num_rooms):
        y[(0, r)] = 0
        y[(1, r)] = 0

    return T_Out, y



def Optimization_control(T_Out, y, initial_temp_room):
    # initial_temp of the room
    # Create the model
    model = LpProblem(name="AC_control", sense=LpMinimize)

    # Initialize the decision variables
    Q_AC = {}
    u = {}
    z = {}
    for r in range(num_rooms):
        for t in range(num_time_intervals):
            Q_AC[(t,r)] = LpVariable(name=f"Q_AC_{t}_{r}")
            u[(t,r)] = LpVariable(name=f"u_{t}_{r}")
            z[(t,r)] = LpVariable(name=f"z_{t}_{r}")
            model += (Q_AC[(t,r)] <= heating_per_5min, f"heat_add_{t}_{r}")
            model += (Q_AC[(t,r)] >= -cooling_per_5min, f"heat_remove_{t}_{r}")
            model += (u[(t,r)] >= Q_AC[(t,r)] * (1.0 / heating_per_5min) * power_usage_heating, f"power_heat_{t}_{r}")
            model += (u[(t, r)] >= Q_AC[(t, r)] * (1.0 / (-cooling_per_5min)) * power_usage_cooling, f"power_cool_{t}_{r}")

    # Temp change across time
    T_room = {}
    delta_T = {}
    Q_Ex = {}


    for r in range(num_rooms):
        for t in range(num_time_intervals):
            T_room[(t, r)] = LpVariable(name=f"T_R_{t}_{r}")
            Q_Ex[(t, r)] = LpVariable(name=f"Q_Ex_{t}_{r}")
            delta_T[(t, r)] = LpVariable(name=f"delta_T_{t}_{r}")
            model += (z[(t, r)] >= T_room[(t, r)] - Target_temp[r], f"z_abs_1_{t}_{r}")
            model += (z[(t, r)] >= -T_room[(t, r)] + Target_temp[r], f"z_abs_2_{t}_{r}")
            if t == 0:
                model += (T_room[(t, r)] == initial_temp_room[r], f"initial_temp_{t}_{r}")
            else:
                model += (Q_Ex[(t-1, r)] == (T_Out[t-1] - T_room[(t-1, r)]) / R[r] * interval_length, f"Q_Ex_{t}_{r}")
                model += (delta_T[(t-1,r)] == (Q_Ex[(t-1, r)] + Q_AC[(t-1,r)]) / (C_air * M_air[r]), f"delta_T_{t}_{r}")
                model += (T_room[(t, r)] == T_room[(t-1, r)] + delta_T[(t-1, r)], f"T_room_{t}_{r}")


    # Add the objective function to the model
    model += lpSum([y[(t,r)]*z[(t,r)] + GAMMA * u[(t,r)] for r in range(num_rooms) for t in range(num_time_intervals)])


    # Solve the problem
    status = model.solve()

    return Q_Ex, Q_AC
    # print(f"status: {model.status}, {LpStatus[model.status]}")
    #
    # print(f"objective: {model.objective.value()}")
    #
    #
    # res = {'room':[], 'time':[],'T_Out':[], 'Q_Ex':[], 'T_room':[], 'delta_T':[], 'num_people_in_class':[], 'power_usage':[], 'Q_AC':[], 'z':[]}
    #
    # for r in range(num_rooms):
    #     for t in range(num_time_intervals):
    #         res['room'].append(r)
    #         res['time'].append(r)
    #         res['T_Out'].append(T_Out[t])
    #         res['T_room'].append(T_room[(t, r)].varValue)
    #         res['Q_Ex'].append(Q_Ex[(t, r)].varValue)
    #         res['delta_T'].append(delta_T[(t, r)].varValue)
    #         res['num_people_in_class'].append(y[(t, r)])
    #         res['Q_AC'].append(Q_AC[(t, r)].varValue)
    #         res['power_usage'].append(u[(t, r)].varValue)
    #         res['z'].append(z[(t, r)].varValue)
    #
    #
    # res_df = pd.DataFrame(res)
    # res_df = res_df.sort_values(['room','time'])
    #
    # res_df.to_csv('results/optimal_solution.csv',index=False)
