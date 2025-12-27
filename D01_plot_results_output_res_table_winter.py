import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from A02_optimization_model import global_target_temp
import numpy as np



def load_data(file_name):
    res_df = pd.read_csv(file_name)
    return res_df


def plot_room_temperature(save_fig, res_list):

    for r in range(num_room):
        time_id = res_list[0].loc[res_list[0]['room'] == r, 'time'].values
        x = time_id
        T_out = res_list[0].loc[res_list[0]['room'] == r, 'T_Out'].values
        num_people = res_list[0].loc[res_list[0]['room'] == r, 'num_people_in_room'].values
        model_name = ['Optimization ($T^{R}_{t,r}$)', 'TTOC ($T^{R}_{t,r}$)', 'TTOCP ($T^{R}_{t,r}$)']
        font_size = 18
        fig, ax1 = plt.subplots(figsize=(15, 8))

        ax2 = ax1.twinx()
        lns1 = []
        idx = 0
        for res in res_list:
            temp = ax1.plot(x, res.loc[res['room'] == r, 'T_room'], lw = 2, color=colors[idx],  label=model_name[idx]) #marker = 's',
            lns1 += temp
            idx += 1
        temp = ax1.plot(x, T_out, color=colors[idx],lw = 2, label='Outside temperature') #marker='s'
        lns1 += temp

        lns2 = ax2.plot(x, num_people, color='k',ls= '--', lw = 2, label = 'Num people in room') #marker='^',

        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]

        time_to_str = {0: '00:00', 48: '04:00', 96: '08: 00',
                       144: '12:00',192:'16:00', 240: '20:00', 288: '24:00'}
        # time_to_str = {0: '00:00', 48: '04:00'}

        x_ticks_time = [key for key in time_to_str]
        x_ticks_str = [value for key, value in time_to_str.items()]
        plt.xticks(x_ticks_time, x_ticks_str)



        # ax1.plot([-1,3],[y1[0],y1[0]],'g--')
        # ax2.plot([-1,3],[y2[0],y2[0]],'b--')

        ax1.set_xlabel('Time of day', fontsize=font_size)
        ax1.set_ylabel('Room temperature ($T^{R}_{t,r}$)', color='k', fontsize=font_size)
        ax2.set_ylabel('Num of people in room', color='k', fontsize=font_size)


        ax1.tick_params(labelsize=font_size)
        ax2.tick_params(labelsize=font_size)
        plt.xticks(fontsize=font_size)

        plt.xlim([-1,289])
        ax1.set_ylim([12, 30])
        if r == 0:
            ax2.set_ylim([0, 30])
        else:
            ax2.set_ylim([0, 30])
        #plt.ylabel('Demand', fontsize=font_size)
        #plt.xlabel( fontsize=font_size)
        # plt.text(-0.1, 1.05, '(d)', fontdict={'size': 18, 'weight': 'bold'},)


        # for i in range(1, len(y1)):
        #     if i == len(y1) - 1:
        #         ax1.text(x[i] - 0.2, y1[i] + 0.7/60, y1_txt_list[i - 1], color='g', fontsize=font_size)
        #     else:
        #         ax1.text(x[i] , y1[i] + 0.8/60, y1_txt_list[i-1], color = 'g',fontsize=font_size)
        #     ax2.text(x[i]- 0.1, y2[i] - 10/60, y2_txt_list[i-1], color = 'b',fontsize=font_size)

        ax1.legend(lns, labs, loc='upper right', fontsize = font_size - 2, ncol = 3)
        plt.tight_layout()
        if save_fig == 1:
            plt.savefig(f'img/model_temp_comparison_room_{r+1}_winter.jpg', dpi=200)
        else:
            plt.show()


def output_res_table(res_list):
    model_name = ['Optimization', 'TTOC', 'TTOCP']
    final_res = []
    for res,model in zip(res_list, model_name):
        res['model'] = model
        final_res.append(res)
    final_res_df = pd.concat(final_res)
    final_res_df['H_rt'] = np.abs(final_res_df['T_room'] - global_target_temp) * final_res_df['num_people_in_room']
    agg_res = final_res_df.groupby(['model'])[['power_usage','H_rt']].sum().reset_index()
    agg_res.to_csv('results/energy_human_comfort_winter.csv',index=False)

if __name__ == '__main__':
    num_room = 2
    colors = sns.color_palette("tab10")
    data_files = ['results/optimal_solution_winter.csv', 'results/on_off_solution_winter.csv',
                  'results/on_off_with_people_solution_winter.csv']
    res = []
    for file in data_files:
        df = load_data(file)
        res.append(df)

    plot_room_temperature(save_fig=1, res_list = res)
    output_res_table(res)