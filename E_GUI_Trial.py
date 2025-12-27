import tkinter
import tkinter.messagebox
import customtkinter
from E02_GUI_model import AC_control_model
from A02_optimization_model import Model_parameters
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.colors = sns.color_palette("tab10")
        self.title("Aircon Optimization Model")
        self.geometry(f"{1400}x{880}")

        #Grid Setup(4X4 with weight=0 considering overfilling frames)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=3)
        self.grid_columnconfigure(3, weight=0)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)

        #Input Parameters as Sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, rowspan=4, column=0, columnspan=2, sticky="nsew")
            #The following configures grid placement inside the sidebar
        self.sidebar_frame.grid_rowconfigure(0, weight=0)
        self.sidebar_frame.grid_rowconfigure(1, weight=1)
        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        self.sidebar_frame.grid_rowconfigure(4, weight=0)
        self.sidebar_frame.grid_columnconfigure(0, weight=0)
        self.sidebar_frame.grid_columnconfigure(1, weight=1)
        self.sidebar_frame.grid_columnconfigure(2, weight=2)
        self.sidebar_frame.grid_columnconfigure(3, weight=1)#add another column 4 with weight=0 for more control if needed
            #Following adds label "Input Parameters" using sidebar grids
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Control Panel", font=customtkinter.CTkFont(size=30, weight="bold"))
        self.logo_label.grid(row=0, column=2, padx=20, pady=(20,0))
            #Run Button
        self.run_button = customtkinter.CTkButton(self.sidebar_frame, text='Run model', command=self.run_model) #MISSING EVENT CODE NEEDS ADDING
        self.run_button.grid(row=3, column=2, pady=0)
            #Appearance Modes
        # self.appearance_mode_optionmenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
        #                                                                command=self.change_appearance_mode_event)
        # self.appearance_mode_optionmenu.grid(row=3, column=0)

        #Tabview Control in Sidebar
        self.tabview = customtkinter.CTkTabview(self.sidebar_frame, width=300)
        self.tabview.grid(row=1, rowspan=2, column=2, padx=10, pady=(10,0), sticky="nsew")
        self.tabview.add("Instructions")
        self.tabview.add("Room 1")
        self.tabview.add("Room 2")
        # self.tabview.add("Room Panel")
            #Individual Tab Gridding
        self.tabview.tab("Instructions").grid_columnconfigure(0, weight=0)
        self.tabview.tab("Instructions").grid_columnconfigure(1, weight=1)
        self.tabview.tab("Instructions").grid_columnconfigure(2, weight=2)
        self.tabview.tab("Instructions").grid_columnconfigure(3, weight=1)
        self.tabview.tab("Instructions").grid_rowconfigure((0,15), weight=0)
        self.tabview.tab("Instructions").grid_rowconfigure((1,2,3,4,5,6,7,8,9,10,11,12,13,14), weight=1)
        self.tabview.tab("Room 1").grid_columnconfigure(0, weight=0)
        self.tabview.tab("Room 1").grid_columnconfigure(1, weight=1)
        self.tabview.tab("Room 1").grid_columnconfigure(2, weight=2)
        self.tabview.tab("Room 1").grid_columnconfigure(3, weight=1)
        self.tabview.tab("Room 1").grid_rowconfigure((0, 15), weight=0)
        self.tabview.tab("Room 1").grid_rowconfigure((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), weight=1)
        self.tabview.tab("Room 2").grid_columnconfigure(0, weight=0)
        self.tabview.tab("Room 2").grid_columnconfigure(1, weight=1)
        self.tabview.tab("Room 2").grid_columnconfigure(2, weight=2)
        self.tabview.tab("Room 2").grid_columnconfigure(3, weight=1)
        self.tabview.tab("Room 2").grid_rowconfigure((0, 15), weight=0)
        self.tabview.tab("Room 2").grid_rowconfigure((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), weight=1)

        #Instruction Panel Setup
        self.Instructions = customtkinter.CTkTextbox(self.tabview.tab("Instructions"), width=350)
        self.Instructions.grid(row=1, rowspan=14, column=2, pady=2.5, sticky="nsew")
        self.Instructions.insert("0.0", "This is an optimization model that utilizes machine learning principles to predict future temperature based on physical thermal modeling. The following is how to operate our model using this interface: \n\n" +
                                 "Final Temperature Data: Input the path of the final temperature CSV file. (From a day up to a month) Check example for specific format.\n\n" +
                                 "Model Control Period: Input period depending on how long you would like to optimize the ACs for into the future. 24 is a day.\n\n" +
                                 "Horizon Interval: Choose a longer horizon interval if the temp-prediction model is accurate based on your case. 12 is suggested.\n\n" +
                                 "Tradeoff Factor: This is the tradeoff between human comfort level and energy conservation. The higher the value, the more emphasis is placed on conserving energy. The lower the value, the more emphasis is placed on human comfort. Default value is suggested for a well balance.\n\n" +
                                 "Room Thermal Resistance: This is the thermal resistance of your room, can be easily calculated based on the wall(s) in contact with the outside world, their surface area & materials.\n\n" +
                                 "Room Air Mass: Calculate by measuring the volume of your room multiplied by air density.\n\n" +
                                 "No.People/Time Data: Path of amount of people in each time interval data file. Check example for specific format.\n\n" +
                                 "AC Heating Power: The heating input of your AC in the specific room, typically given by the AC specs on the specific AC.\n\n" +
                                 "AC Cooling Power: The cooling input of your AC in the specific room, typically given by the AC specs on the specific AC.\n\n" +
                                 "Energy Efficiency Ratio: Otherwise known as the coefficient of performance, given by the AC specs.")

        #Input Parameters Perimeters Setup Room 1
            # Model Parameters Room 1
        self.FinalTempCSVFilePath_Input = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.FinalTempCSVFilePath_Input.insert(0, "data/temperature_final_5_min.csv")
        self.FinalTempCSVFilePath_Input.grid(row=1, column=3, pady=2.5)

        self.Modelvalue_control_period = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.Modelvalue_control_period.insert(0, "24")
        self.Modelvalue_control_period.grid(row=2, column=3, pady=2.5)

        self.NumHorizonInterval = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.NumHorizonInterval.insert(0, "12")
        self.NumHorizonInterval.grid(row=3, column=3, pady=2.5)

        self.TradeoffFactor_Gamma = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.TradeoffFactor_Gamma.insert(0, "1e-3")
        self.TradeoffFactor_Gamma.grid(row=4, column=3, pady=2.5)
            # Room Parameters Room 1
        self.RoomThermalResistance = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.RoomThermalResistance.insert(0, '0.036')
        self.RoomThermalResistance.grid(row=5, column=3, pady=2.5)

        self.RoomAirMass = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.RoomAirMass.insert(0, '312.4')
        self.RoomAirMass.grid(row=6, column=3, pady=2.5)

        self.NumberPeopleInRoom = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.NumberPeopleInRoom.insert(0, "data/num_people_in_room.xlsx")
        self.NumberPeopleInRoom.grid(row=7, column=3, pady=2.5)
            #AC Parameters Room 1
        self.ACHeatingPower = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.ACHeatingPower.insert(0, "1100")
        self.ACHeatingPower.grid(row=8, column=3, pady=2.5)

        self.ACCoolingPower = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.ACCoolingPower.insert(0, "1000")
        self.ACCoolingPower.grid(row=9, column=3, pady=2.5)

        self.EnergyEfficiencyRatio = customtkinter.CTkEntry(self.tabview.tab("Room 1"), width=200)
        self.EnergyEfficiencyRatio.insert(0, "2.75")
        self.EnergyEfficiencyRatio.grid(row=10, column=3, pady=2.5)

        # self.ACvalue_4 = customtkinter.CTkComboBox(self.tabview.tab("Input Parameters"),
        #                                              values=["Value 4", "..."], width=180)
        # self.ACvalue_4.grid(row=4, column=3, pady=7)
        # self.ACvalue_5 = customtkinter.CTkComboBox(self.tabview.tab("Input Parameters"),
        #                                            values=["Value 5", "..."], width=180)
        # self.ACvalue_5.grid(row=5, column=3, pady=7)
        # self.ACvalue_6 = customtkinter.CTkComboBox(self.tabview.tab("Input Parameters"),
        #                                            values=["Value 6", "..."], width=180)
        # self.ACvalue_6.grid(row=6, column=3, pady=7)

        # self.master_popup_window.grid(row=14, column=1, columnspan=3, pady=10)

        # Model Parameters Room 2
        self.FinalTempCSVFilePath_Input = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.FinalTempCSVFilePath_Input.insert(0, "data/temperature_final_5_min.csv")
        self.FinalTempCSVFilePath_Input.grid(row=1, column=3, pady=2.5)

        self.Modelvalue_control_period = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.Modelvalue_control_period.insert(0, "24")
        self.Modelvalue_control_period.grid(row=2, column=3, pady=2.5)

        self.NumHorizonInterval = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.NumHorizonInterval.insert(0, "12")
        self.NumHorizonInterval.grid(row=3, column=3, pady=2.5)

        self.TradeoffFactor_Gamma = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.TradeoffFactor_Gamma.insert(0, "1e-3")
        self.TradeoffFactor_Gamma.grid(row=4, column=3, pady=2.5)
        # Room Parameters Room 2
        self.RoomThermalResistance = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.RoomThermalResistance.insert(0, '0.031')
        self.RoomThermalResistance.grid(row=5, column=3, pady=2.5)

        self.RoomAirMass = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.RoomAirMass.insert(0, '165.4')
        self.RoomAirMass.grid(row=6, column=3, pady=2.5)

        self.NumberPeopleInRoom = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.NumberPeopleInRoom.insert(0, "data/num_people_in_classroom.xlsx")
        self.NumberPeopleInRoom.grid(row=7, column=3, pady=2.5)
        # AC Parameters Room 2
        self.ACHeatingPower = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.ACHeatingPower.insert(0, "1100")
        self.ACHeatingPower.grid(row=8, column=3, pady=2.5)

        self.ACCoolingPower = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.ACCoolingPower.insert(0, "1000")
        self.ACCoolingPower.grid(row=9, column=3, pady=2.5)

        self.EnergyEfficiencyRatio = customtkinter.CTkEntry(self.tabview.tab("Room 2"), width=200)
        self.EnergyEfficiencyRatio.insert(0, "2.75")
        self.EnergyEfficiencyRatio.grid(row=10, column=3, pady=2.5)

        # self.master_popup_window.grid(row=14, column=1, columnspan=3, pady=10)

        #Panel Setups
        #Room 1 Panel
        self.finaltemperaturedata_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"), text="Final Temperature Data",
                                                 font=customtkinter.CTkFont(size=12))
        self.finaltemperaturedata_label.grid(row=1, column=2, pady=2.5)

        self.controlperiod_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="Model Control Period",
                                                            font=customtkinter.CTkFont(size=12))
        self.controlperiod_label.grid(row=2, column=2, pady=2.5)

        self.horizoninterval_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="No. Horizon Interval",
                                                            font=customtkinter.CTkFont(size=12))
        self.horizoninterval_label.grid(row=3, column=2, pady=2.5)

        self.tradeofffactor_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="Tradeoff Factor",
                                                            font=customtkinter.CTkFont(size=12))
        self.tradeofffactor_label.grid(row=4, column=2, pady=2.5)

        self.thermalresistance_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="Room Thermal Resistance",
                                                            font=customtkinter.CTkFont(size=12))
        self.thermalresistance_label.grid(row=5, column=2, pady=2.5)

        self.roomairmass_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                        text="Room Air Mass",
                                                        font=customtkinter.CTkFont(size=12))
        self.roomairmass_label.grid(row=6, column=2, pady=2.5)

        self.no_people_time_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="No. People/Time Data",
                                                            font=customtkinter.CTkFont(size=12))
        self.no_people_time_label.grid(row=7, column=2, pady=2.5)

        self.acheatingpower_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="AC Heating Power",
                                                            font=customtkinter.CTkFont(size=12))
        self.acheatingpower_label.grid(row=8, column=2, pady=2.5)

        self.accoolingpower_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="AC Cooling Power",
                                                            font=customtkinter.CTkFont(size=12))
        self.accoolingpower_label.grid(row=9, column=2, pady=2.5)

        self.energyefficiencyratio_label = customtkinter.CTkLabel(self.tabview.tab("Room 1"),
                                                            text="Energy Efficiency Ratio",
                                                            font=customtkinter.CTkFont(size=12))
        self.energyefficiencyratio_label.grid(row=10, column=2, pady=2.5)

        # Room 2 Panel
        self.finaltemperaturedata_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                                 text="Final Temperature Data",
                                                                 font=customtkinter.CTkFont(size=12))
        self.finaltemperaturedata_label.grid(row=1, column=2, pady=2.5)

        self.controlperiod_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                          text="Model Control Period",
                                                          font=customtkinter.CTkFont(size=12))
        self.controlperiod_label.grid(row=2, column=2, pady=2.5)

        self.horizoninterval_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                            text="No. Horizon Interval",
                                                            font=customtkinter.CTkFont(size=12))
        self.horizoninterval_label.grid(row=3, column=2, pady=2.5)

        self.tradeofffactor_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                           text="Tradeoff Factor",
                                                           font=customtkinter.CTkFont(size=12))
        self.tradeofffactor_label.grid(row=4, column=2, pady=2.5)

        self.thermalresistance_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                              text="Room Thermal Resistance",
                                                              font=customtkinter.CTkFont(size=12))
        self.thermalresistance_label.grid(row=5, column=2, pady=2.5)

        self.roomairmass_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                        text="Room Air Mass",
                                                        font=customtkinter.CTkFont(size=12))
        self.roomairmass_label.grid(row=6, column=2, pady=2.5)

        self.no_people_time_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                           text="No. People/Time Data",
                                                           font=customtkinter.CTkFont(size=12))
        self.no_people_time_label.grid(row=7, column=2, pady=2.5)

        self.acheatingpower_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                           text="AC Heating Power",
                                                           font=customtkinter.CTkFont(size=12))
        self.acheatingpower_label.grid(row=8, column=2, pady=2.5)

        self.accoolingpower_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                           text="AC Cooling Power",
                                                           font=customtkinter.CTkFont(size=12))
        self.accoolingpower_label.grid(row=9, column=2, pady=2.5)

        self.energyefficiencyratio_label = customtkinter.CTkLabel(self.tabview.tab("Room 2"),
                                                                  text="Energy Efficiency Ratio",
                                                                  font=customtkinter.CTkFont(size=12))
        self.energyefficiencyratio_label.grid(row=10, column=2, pady=2.5)


        #Right_Frame: Slider and Progressbar
        self.right_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, rowspan=4, column=2, columnspan=4, sticky="nsew")
        self.right_frame.grid_columnconfigure((0,6), weight=0)
        self.right_frame.grid_columnconfigure((1,2,3,4,5), weight=1)
        self.right_frame.grid_rowconfigure((0, 8), weight=0)
        self.right_frame.grid_rowconfigure((1,2,3,4,5,6,7), weight=1)
        self.progressbar = customtkinter.CTkProgressBar(self.right_frame, orientation="horizontal", mode="determinate", width=400)
        self.progressbar.grid(row=8, column=1, columnspan=5, pady=(0, 50))
        self.progressbar.set(0)

        # Plot_Tabview
        self.plot_tabview = customtkinter.CTkTabview(self.right_frame, width=500)
        self.plot_tabview.grid(row=1, rowspan=6, column=1, columnspan=5, padx=10, pady=(0, 0), sticky="nsew")
        self.plot_tabview.add("Room 1 Results Plotting")
        self.plot_tabview.add("Room 2 Results Plotting")
        # self.tabview.add("Room 2")
        # Individual Tab Gridding
        self.plot_tabview.tab("Room 1 Results Plotting").grid_columnconfigure(0, weight=0)
        self.plot_tabview.tab("Room 1 Results Plotting").grid_columnconfigure(1, weight=1)
        self.plot_tabview.tab("Room 1 Results Plotting").grid_columnconfigure(2, weight=2)
        self.plot_tabview.tab("Room 1 Results Plotting").grid_columnconfigure(3, weight=1)
        self.plot_tabview.tab("Room 1 Results Plotting").grid_rowconfigure((0, 15), weight=0)
        self.plot_tabview.tab("Room 1 Results Plotting").grid_rowconfigure((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), weight=1)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_columnconfigure(0, weight=0)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_columnconfigure(1, weight=1)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_columnconfigure(2, weight=2)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_columnconfigure(3, weight=1)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_rowconfigure((0, 15), weight=0)
        self.plot_tabview.tab("Room 2 Results Plotting").grid_rowconfigure((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), weight=1)




    #Event Definitions
    def change_appearance_mode_event(self, new_appearance_mode: str):
            customtkinter.set_appearance_mode(new_appearance_mode)

    def run_model(self):

        # revise & add parameters:
        input_ACHeatingPower = int(self.ACHeatingPower.get())
        input_ACCoolingPower = int(self.ACCoolingPower.get())
        input_energy_efficiency_ratio = float(self.EnergyEfficiencyRatio.get())
        input_data_temp_file = self.FinalTempCSVFilePath_Input.get()
        input_control_period_hr = int(self.Modelvalue_control_period.get())
        input_num_horizon_interval = int(self.NumHorizonInterval.get())
        input_tradeoff_gamma = float(self.TradeoffFactor_Gamma.get())
        input_num_people_in_classroom_file = self.NumberPeopleInRoom.get()
        input_room_thermal_resistance = self.RoomThermalResistance.get()
        input_room_air_mass = self.RoomAirMass.get()

        model_para = Model_parameters(AC_power_heating=input_ACHeatingPower,
                                      AC_power_cooling=input_ACCoolingPower,
                                      energy_efficiency_ratio=input_energy_efficiency_ratio,
                                      interval_length_min=5,
                                      num_horizon_intervals=input_num_horizon_interval,
                                      gamma=input_tradeoff_gamma,
                                      room_air_mass=input_room_air_mass,
                                      room_thermal_resistance=input_room_thermal_resistance)
        ####
        self.model_para = model_para
        num_time_idx = input_control_period_hr * 60 // model_para.interval_length_min
        model = AC_control_model(data_temp_file=input_data_temp_file,
                                 num_people_in_classroom_file=input_num_people_in_classroom_file,
                                 control_period_hr=input_control_period_hr,
                                 model_para=model_para)
        Q_Ex, Q_AC = None, None


        heating = {'t':[],'Q':[]}
        cooling = {'t':[],'Q':[]}
        r = 0 # room_id
        eps = 1e-5
        max_Q_AC_cooling = self.model_para.cooling_per_5min
        max_Q_AC_heating = self.model_para.heating_per_5min

        text_var = tkinter.StringVar(value=f"Room Energy Consumed: {round(0, 2)} kW",)
        self.final_energy_display = customtkinter.CTkLabel(self.right_frame,
                                                           textvariable=text_var,
                                                           font=customtkinter.CTkFont(size=15, weight="bold"))
        self.final_energy_display.grid(row=7, column=2, columnspan=3)
        font_size = 18
        for t in range(num_time_idx):
            self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
            self.ax2 = self.ax1.twinx()
            Q_Ex, Q_AC, res = model.run_model_step(t, Q_Ex, Q_AC)
            res_df = pd.DataFrame(res)
            res_df_room = res_df.loc[res_df['room'] == r]
            Q_AC_list = res_df_room['Q_AC'].values
            if Q_AC_list[t] > 0 + eps:
                heating['t'].append(t)
                heating['Q'].append(Q_AC_list[t] / max_Q_AC_heating * 0.7 + 0.15)
            if Q_AC_list[t] < 0 - eps:
                cooling['t'].append(t)
                cooling['Q'].append(Q_AC_list[t] / max_Q_AC_cooling * 0.7 + 0.15)
            self.progressbar.set(t/num_time_idx)
            self.plot_control_result(res_df_room, heating, cooling, r)
            # Energy Consumed
            energy_used = sum(res_df_room['power_usage']) / 1000
            text_var.set(f"Room Energy Consumed: {round(energy_used, 2)} kW")
            self.update()
            plt.close()
            # self.ax2.clear()
            # self.ax2 = self.ax1.twinx()

    def plot_control_result(self, res_df, heating, cooling, r):

        res_list = [res_df]
        time_id = res_list[0].loc[res_list[0]['room'] == r, 'time'].values


        x = time_id
        T_out = res_list[0].loc[res_list[0]['room'] == r, 'T_Out'].values
        num_people = res_list[0].loc[res_list[0]['room'] == r, 'num_people_in_room'].values
        model_name = ['Room temperature']
        font_size = 18
        T_max = max(T_out)
        T_min = min(T_out)
        N_max = max(num_people)

        lns1 = []
        idx = 0
        for res in res_list:
            temp = self.ax1.plot(x, res.loc[res['room'] == r, 'T_room'].values, lw = 2, color=self.colors[idx],  label=model_name[idx]) #marker = 's',
            lns1 += temp
            idx += 1
            if max(res.loc[res['room'] == r, 'T_room']) > T_max:
                T_max = max(res.loc[res['room'] == r, 'T_room'])
            if min(res.loc[res['room'] == r, 'T_room']) < T_min:
                T_min = min(res.loc[res['room'] == r, 'T_room'])

        temp = self.ax1.plot(x, T_out, color=self.colors[idx],lw = 2, label='Outside temperature') #marker='s'
        lns1 += temp



        lns2 = self.ax2.plot(x, num_people, color='k',ls= '--', lw = 2, label = 'Num people in room') #marker='^',



        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]

        # print(cooling)
        # print(heating)

        if cooling['t']:
            cool = self.ax1.scatter(cooling['t'], [T_max * 1.01] * len(cooling['t']), s=50, c=cooling['Q'], cmap = 'Blues', alpha=0.2, label='AC Cooling')
            labs += [cool.get_label()]
        if heating['t']:
            heat = self.ax1.scatter(heating['t'], [T_max * 1.01] * len(heating['t']), s=50, c=heating['Q'], cmap = 'Reds', alpha=0.2, label='AC Heating')
            labs += [heat.get_label()]


        input_control_period_hr = int(self.Modelvalue_control_period.get())
        num_time_idx = input_control_period_hr * 60 // self.model_para.interval_length_min
        time_to_str = {0: '00:00', 48: '04:00', 96: '08: 00',
                       144: '12:00',192:'16:00', 240: '20:00', 288: '24:00'}

        cur_index = 288
        while True:
            if num_time_idx <= cur_index:
                break
            cur_index += 48
            if cur_index - 288 - 48 in time_to_str:
                time_to_str[cur_index] = time_to_str[cur_index - 288 - 48]



        # time_to_str = {0: '00:00', 48: '04:00'}

        x_ticks_time = [key for key in time_to_str]
        x_ticks_str = [value for key, value in time_to_str.items()]
        plt.xticks(x_ticks_time, x_ticks_str)
        self.ax1.set_xlabel('Time of day', fontsize=font_size)
        self.ax1.set_ylabel('Temperature', color='k', fontsize=font_size)
        self.ax2.set_ylabel('Num of people in room', color='k', fontsize=font_size)

        self.ax1.tick_params(labelsize=font_size)
        self.ax2.tick_params(labelsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.xlim([-1,num_time_idx+1])
        self.ax1.set_ylim([T_min*0.95, T_max*1.1])
        if N_max <= 5:
            self.ax2.set_ylim([0, 5])
        else:
            self.ax2.set_ylim([0, N_max*1.5])

        lines, labels = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()

        self.ax1.legend(lines + lines2, labels + labels2, loc='upper right', fontsize = font_size - 2, ncol = 2)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(self.fig, master=self.plot_tabview.tab("Room 1 Results Plotting"))
        canvas.draw()
        canvas.get_tk_widget().grid(row=7, column=1, columnspan=4)


    # def change_progress_bar(self):
    #       self.progressbar.start()
        

if __name__ == "__main__":
    app = App()
    app.mainloop()


