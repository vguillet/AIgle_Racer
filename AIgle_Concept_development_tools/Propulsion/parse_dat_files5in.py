import os
import numpy as np
import pandas as pd
import re
from io import StringIO
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as m 
#scikit for regression
#%%################################################----------------- parameters (LIMITS) --------------------------###########################################
    ##### GENERAL #####
    #Theres a total of 520 propeller dat files: 5 to 7 inch are between range(343,417)
    #Vmph_req = 49.77 mph
    #Thrust total per psu =  2.4129594596135115 lbf
dat_start = 343         # DO NOT TOUCH - this means it only starts looking from 5 inch props onwards
dat_end = 361           # DO NOT TOUCH - end of 5 inch
#dat_start = 361         # DO NOT TOUCH - start 6 inch
#dat_end = 369           # DO NOT TOUCH - end 6 inch
    # 5 inch starts at index 343
    # 6 inch starts at index 367
    # 7 inch starts at index 385
    # 7 inch ends at index 416 so use 417
mass_estimate = 1100                                                # CONCEPT MASS ESTIMATE [g]
T_W_REQ = 4                                                      # Required thrust to weight ratio
T_W_toplim = 4.5                                                    #Top limit for thrust to weight ratio - ensure this is higher than the required!
n_motors = 4                                                        # amount of motors
v_free = 10.00                                                      # free stream velocity [m/s]
max_tilt = 75                                                       # max tilt angle [degrees]
v_axial = m.cos(m.radians(90-max_tilt))*v_free                      # speed hitting the propeller - axial velocity [m/s]
v_margin_mph = 2                                                    # margin in mile-per-hour [mph]

    ### bottom & top limits computations ###
v_axial_mph = 2.23694*v_axial                                       # speed hitting the propeller - axial velocity [mph]
v_axial_mph_top = v_axial_mph+v_margin_mph                          # TOP LIMIT speed hitting the propeller - axial velocity [mph] 
v_axial_mph_bot = v_axial_mph-v_margin_mph                          # bottom LIMIT speed hitting the propeller - axial velocity [mph] 
v_axial_top = v_axial_mph_top/2.23694                               # TOP LIMIT speed hitting the propeller - axial velocity [m/s] 
v_axial_bot = v_axial_mph_bot/2.23694                               # bottom LIMIT speed hitting the propeller - axial velocity [m/s] 
gram_to_lbf = 0.0022046226218488                                    # conversion from gram to pound force
thrust_per_m_gram = (mass_estimate*T_W_REQ)/n_motors                # thrust per motor required (bottom limit) in grams
thrust_per_m_lbf = thrust_per_m_gram*gram_to_lbf     # thrust per motor required (bottom limit) in grams
thrust_top_m_gram = (mass_estimate*T_W_toplim)/n_motors             # top limit thrust amount generated [g] - ensure this higher than bottom limit of thrust!
thrust_top_m_lbf = gram_to_lbf*thrust_top_m_gram                    # top limit thrust amount generated [lbf]
##############------- PROPELLER DATABASE RELATED ---------############### 
    #### PROP FILTER 1 PARAMETERS ####
Vbot = v_axial_mph_bot                                   # BOTTOM LIMIT OF AXIAL VELOCITY [MPH]   - if this doesnt matter choose a very low value!
Vtop = v_axial_mph_top                                   # TOP LIMIT OF AXIAL VELOCITY [MPH]      - if this doesnt matter choose a very high value!
Tbot = thrust_per_m_lbf                                  # BOTTOM LIMIT OF TOTAL THRUST per psu [LBF] , t/w = 3.98   - if this doesnt matter choose a very low value!
Ttop = thrust_top_m_lbf                                  # TOP LIMIT OF TOTAL THRUST per psu [LBF], t/w = 3.98   - if this doesnt matter choose a very high value!
#hover Thrust per psu =  0.6062712210084201 lbf
#Tbot_hov = 0.61                             # BOTTOM LIMIT OF hover THRUST per psu [LBF], t/w = 1    - if this doesnt matter choose a very low value!
Tbot_hov = thrust_per_m_lbf # uses bottom thrust limit                # BOTTOM LIMIT OF hover THRUST per psu [LBF], t/w = 1    - if this doesnt matter choose a very low value!
Ttop_hov = 15                               # TOP LIMIT OF hover per psu [LBF], t/w = 1  - if this doesnt matter choose a very high value!
prop_number = 15                            # number of propellers used for analysis (4 means it will select the top 4 props from the dat name file)
#prop_number = 520 # uses ALL props - based on line 61 

### should Tbot_hov as Tbot


    #### PROP FILTER 2 PARAMETERS ####
com_rpm_bot = 0                            # bottom RPM limit used for FILTER 2  - if this doesnt matter choose a very low value!
com_rpm_top = 100000                         # top RPM limit used for FILTER 2 - if this doesnt matter choose a very high value!
com_tw_bot = 1.5                           # bottom t/w limit used for FILTER 2  - if this doesnt matter choose a very low value!
com_tw_top = 10                             # top t/w limit used for FILTER 2 - if this doesnt matter choose a very high value!

################## -----END PROP RELATED-------#############################


##############------- MOTOR DATABASE RELATED ---------###############

    # ASSUMPTIONS 
eta_motor = 0.80                            # electric motor ASSUMED efficiency
    # MOTOR FILTER 1 PARAMETERS
thrust_factor_margin = 0.0 # thrust factor margin for the dynamic-static conversion
# CURRENTLY ONLY FACTOR A IS USED NOT B
m_thrust_bot = thrust_per_m_gram    # bottom limit motor DB THRUST [g]   - if this doesnt matter choose a very low value!
m_thrust_top = thrust_top_m_gram    # top limit motor DB THRUST [g]  - if this doesnt matter choose a very high value!
m_tw_bot = 1.5        # bottom limit motor DB t/w [g/W]    - if this doesnt matter choose a very low value!
m_tw_top = 10          # top limit motor DB t/w [g/W]   - if this doesnt matter choose a very high value!
m_rpm_bot = 0          # bottom limit motor DB RPM          - if this doesnt matter choose a very low value!
m_rpm_top = 100000      # top limit motor DB RPM         - if this doesnt matter choose a very high value!
    # MOTOR FILTER 2 PARAMETERS
# GO TO SECTION AND IMPLEMENT CORRECT IF STATEMENT IF YOU WANT 6 INCH OR 5 INCH PROPELLERS ONLY
# filter two chooses winning propeller for every motor
    ### motor names to be skipped, because they cant be inputted into ECALC ###
# for 6 inch: cobra champion,RCTimer FR2205 2300kv  skipped due to overheat
#motorskiplist = ['RacerStar BR2205 2600kv','ZTW Spider 2205 2300kv','DJI Snail 2305 2400kv','ZMX 2205 2300kv','RCX SE2205 2400kv','LDPower 2305 2300kv','Leopard Hobby PHS2205 2300kv','DYS Sun Fun 2207 2400kv','Cobra Champion 2205 2300kv','RCTimer FR2205 2300kv'] # 6 inch -motor names to be skipped
# for 5 inch: EMAX RS2205S skipped due to overheating
motorskiplist = ['Emax RS2205S 2600kv','Leopard Hobby PHS2205 2300kv','FPVAces X4pro 2207 2350kv','RCX RS2206v2 2400kv','Multicopter Builders 2207 2400kv','Multicopter Builders Primo 2207 2450kv','SkyGear SG2207 2400kv','DroneBuilds Demon 2207 2450kv','Dragonfly Hurricane 2207 2650kv','Brother Hobby R5 2306 2450kv','Brother Hobby R3 2207 1660kv','DroneBuilds Demon 2207 2450kv','Multicopter Builders Primo 2208 2350kv','BrotherHobby Sphinx FS2405 2425kv','RMRC Rifle 2206 2300kv','ZDrone Raptor 2206 2535kv','GepRC SpeedX GR2306 2450kv','Northaero Dragon 2207.5 2650kv','AMAXInno 2305 2350kv','AMAXinno 2305 2550kv','AMAXInno 2306 2500kv','AMAXInno 2307 2500kv','X-Foot 2207 2600kv','RMRC Rifle 2206 2550kv','ZMX IJN Umikaze FinX25','ZMX IJN Umikaze FinX24','ZMX IJN Umikaze FinX23','AMAXInno 2207.5 2500kv','BeeRotor ZOE Z2207 2780kv','3B-R 2206 2500kv','Hypetrain Botgrinder 2306 2450kv','Rebel Pro 2206(6.5) 2300kv','Multicopter Builders 2206 2400kv','Armattan 2206 2300kv Prototype','HaloRC Icarus 2205 2500kv','Hyperlite Team 2206 2700kv','Lumenier "JohnnyFPV" 2207 2700kv','iFlight Force IF2207 2700kv','iPower IX2205M 2500kv','RacerStar RF2205S 2300kv','Sumax Innovation 2207 2600kv','360 Hobbies Orbit 2205 2550kv','RCINPower GTS2305 2700kv','Northaero Dragon 2207.5 2450kv','AOKFly FR2205Pro 2650kv','ZMXv2 2206 2600kv','SunnySky Edge Lite R2305 2480kv','Sumax SR2206B 2600kv','ZMX Fusion X20 2205 2300kv'] # 5 inch - motorskiplist 
################## -----END MOTOR RELATED-------#############################
  
    
################## -----BEGIN MATCHING & BATTERY RELATED-------#############################
Minimum_MAXdistance = 0.50 # absolute minimum of maximum distance between two points FOR MATCHING! - use for 5 inch -> use a low number
rpm_margin = 600 # sets rpm margin when margin
#Minimum_distance =  10 # absolute minimum distance between two points FOR MATCHING! - use for 6 inch
# for 6 inch: change if statement in motor filt2 section 
# for 6 inch: disable if statement (yDis) in matching

    ### BATTERY RELATED ###
FT = 60                         # flight time [seconds]
FT_idle = 120                   # idle time no motors active only dry current
I_dry = 2                    # Dry current [Amps]            
c_rate_per = 20/100             # c-rate percentage on difference - input in decimals!
c_rate_max = 100                  # minimum c-rate
on_off1 = 1                     # switch on or off battery filter (filters on current & capacity requirement, c-rate NOT MOTOR WEIGHT!)
max_Bweight = 500               # MAX battery weight
eta_bat = 0.80                  # efficiency of the BATTERY - IS THIS DOD? or are there more

# for more info on matching go to that section's header


################## -----END MATCHING & BATTERY RELATED-------#############################
    ###notes: 
# check if efficiencies are taken everywhere
# should the props & motors not only have the same t/pwr but also the same thrust? 
# SHOULD WE USE THE DYNAMIC POWER OF STATIC POWER FOR THE PROPELLER T/PWR
    # MOTOR
    # MATCHING
# ensure motor or propeller t/pwr is chosen?? for matching??
 

    ##### CONVERSIONS & MISC # DO NOT TOUCH #######
hp_to_pwr = 745.699872                      # horse power to watt
lbf_to_g = 453.5927                         # pound force to grams
print('Thrust to weight inputted:',T_W_REQ ,' up to ',T_W_toplim)
print('Free-stream velocity inputted:',v_free ,'[m/s] //  Max tilt angle:',max_tilt, 'degrees')
print('flight time (@max) inputted:',FT ,'seconds // dry current:', I_dry,' Amps // IDLE flight time:',FT_idle,' seconds')
print('Filter will use following limits:')
print('Thrust per motor (dynamic): ',thrust_per_m_gram,' up to ',thrust_top_m_gram, 'grams')
print('Axial velocity: ',round(v_axial_bot,2),' up to ',round(v_axial_top,2), '[m/s]', '// actual v_axial =',round(v_axial,2),'// mph margin chosen:', v_margin_mph)
print('------END OF PARAMETER DETERMINATION SECTION -------')
#################################################----------------- parameters (LIMITS) END ----------------------###########################################
#%% DEFINITION TO OBTAIN PROPELLER DAT FILE

def DAT_PROP(DATFILE,Vbot,Vtop,Tbot,Ttop):
    DAT_FILE_NAME = DATFILE
    with open(DAT_FILE_NAME) as f:
        f_contents = f.read()

    # split text based on PROP RPM =       XXXX line
    RPM_values = re.findall(r'^ +PROP RPM =\s+(\d+) +\n', f_contents, re.MULTILINE)
    split_strings = re.split(r'^ +PROP RPM =\s+\d+ +\n', f_contents, 0, re.MULTILINE)
    # remove first txt
    split_strings.pop(0)

    # remove empty lines from strings
    split_strings = [os.linesep.join([s for s in text.splitlines() if s.strip()])
                 for text in split_strings]
    # remove units lines
    split_strings = [os.linesep.join([x for i, x in enumerate(text.splitlines()) if i != 1])
                 for text in split_strings]

    data_list = []
    for (s, rpm) in zip(split_strings, RPM_values):
        df_tmp = pd.read_fwf(StringIO(s), usecols=[
                "V", "J", "Pe", "Ct", "Cp", "PWR", "Torque", "Thrust"])
        df_tmp['RPM'] = float(rpm)
        data_list.append(df_tmp)
        
    df_main = pd.concat(data_list, ignore_index=True)
    df_new = pd.DataFrame(columns=df_main.columns)
    V = df_main["V"]
    T = df_main["Thrust"]
    RPM = df_main["RPM"]
    for i in range(0,len(V)): # FILTER 1
        if  V[i] > Vbot and V[i] < Vtop and T[i] > Tbot and T[i] < Ttop: # hard requirements in flight conditions
            rpm_thres = df_main["RPM"][i]
            for j in range(0,len(V)):           # needed for static conditions
                if V[j] == 0 and RPM[j] == rpm_thres and T[j] > Tbot_hov:       # add v = 0 row if True value is found for that value's rpm
                    df_vzero = df_main[j:j+1]
                    df_new = pd.concat([df_new,df_vzero])
            df_row = df_main[i:i+1]
            df_new = pd.concat([df_new,df_row]) # adds dataframe row to new (o.g. empty) dataframe    
    df_new = df_new.reset_index(drop= True)
    return (df_new)
#%% CREATING PROPELLER DICTIONARY ###
### INCLUDES FILTER 1 ###
df_titles = pd.read_csv('PER2_TITLEDATcc.dat',skiprows = 3,delim_whitespace = True)
DATtitles = df_titles["PER3_105x45.dat"]    # name of dat file of prop - (uses this name since column is named like this!)
NO_DATtitles = df_titles["10.5x4.5"]        # name of actual prop - (uses this name since column is named like this!)
PROP_dct = {}                              # key = PROP name / Value = DATAFRAME of all parameters
filtered1_list = []
for i in range(dat_start,dat_end):
    PROP_dat = DAT_PROP(DATtitles[i],Vbot,Vtop,Tbot,Ttop) 
    if len(PROP_dat['V']) > 0: # NEGATES EMPTY DATAFRAMES
        PROP_dct[NO_DATtitles[i]] = PROP_dat
        filtered1_list.append(NO_DATtitles[i])
#print('Amount of propellers in prop dat file = ',len(DATtitles))
print('OG list of',dat_end-dat_start,' props filtered to',len(filtered1_list), 'props by filter_1')
print('--------DONE WITH PROP DICTIONARY-------- ')
#%% USING PROPELLER DICTIONARY TO ACCESS VALUES ###
### INCORPORATION OF FILTER 2 ####
### changing from dynamic to static case -- ENTIRE DATA

PROPFILT_dct = {}
filtered2_list = []
propMATCH_df = pd.DataFrame(columns = ['NAME','PWR[W]','Thrust[g]','T/W[g/W]','RPM[rev/s]']) # OG dataframe used for PROP comparison and STORAGE!
#propMATCH_df is used later for the prop-motor matching plot
#propcomplot_df is used directly in next section to create plot
thrustdif_df = pd.DataFrame(columns = ['Thrust factor'])    # dataframe to collect differences in dynamic and static thrust for propellers passing filter
for j in range(0,len(filtered1_list)):
    prop_title = filtered1_list[j]               # sets name - used for label
    prop_df = PROP_dct[prop_title]        # selects dataframe
    T1 = prop_df['Thrust']
    PWR1 = prop_df['PWR']
    RPM1 = prop_df['RPM']
    V1 = prop_df['V']
    propcomplot_df = pd.DataFrame(columns = ['NAME','PWR[W]','Thrust[g]','T/W[g/W]','RPM[rev/s]']) # datafram used for comparison plots, RESET contantly!
    for i in range(0,len(PWR1)):
        if V1[i] == 0:
            thrusti = T1[i]*lbf_to_g                  # thrust from v = 0 used 
            poweri = PWR1[i+1]*hp_to_pwr/eta_motor    #power used from v = v_free! (=i+1) + assummed efficiency/ if using v_static use  (=i) 
            T_Wi = thrusti/poweri                     # Thrust to power ratio 
            thrustdifi = T1[i]/T1[i+1]                # difference in thrust between dynamic and static condition
            if T_Wi > com_tw_bot and RPM1[i] > com_rpm_bot and RPM1[i] < com_rpm_top and T_Wi < m_tw_top:
                T_PWR = pd.DataFrame([[prop_title,poweri,thrusti,T_Wi,RPM1[i]]], columns= ['NAME','PWR[W]','Thrust[g]','T/W[g/W]','RPM[rev/s]']) # uses power from 
                propcomplot_df = pd.concat([propcomplot_df,T_PWR], ignore_index = True)
                propMATCH_df = pd.concat([propMATCH_df,T_PWR], ignore_index = True)
                thrustdif_dfi = pd.DataFrame([[thrustdifi]], columns = ['Thrust factor'])
                thrustdif_df = pd.concat([thrustdif_df,thrustdif_dfi], ignore_index = True)
    if len(propcomplot_df['PWR[W]']) > 0: # NEGATES EMPTY DATAFRAMES
        PROPFILT_dct[prop_title] = propcomplot_df
        filtered2_list.append(prop_title)
thrust_factor_max = thrustdif_df.max() # MAX thrust factor (static/dynamic)
thrust_factor_min = thrustdif_df.min() # MIN thrust factor (static/dynamic)
thrust_factor_avg = thrustdif_df.mean() # AVG thrust factor (static/dynamic)
print('filter_1 list of',len(filtered1_list),' props filtered to',len(filtered2_list), 'props by filter 2')  
print('MAX thrust factor (static/dynamic) = ',round(float(thrust_factor_max),3),'// MIN value = ',round(float(thrust_factor_min),3),'// AVG val =',round(float(thrust_factor_avg),3) )
print('--------DONE WITH FILTERED(2) PROP DICTIONARY-------- ')

#%% PLOTTING FILTERED(2) PROPELLER DATA

fig12 = plt.figure('T/W vs RPM (PROPS - FILTERED(2))')
fig12.clf()
ax_motor12 = plt.subplot(111)
for i in range(0,len(filtered2_list)):
    prop_title = filtered2_list[i] # used for label and accessing dictionary
    prop_df =  PROPFILT_dct[prop_title]
    plt.scatter(prop_df['RPM[rev/s]'],prop_df['T/W[g/W]'], label = (prop_title))
box = ax_motor12.get_position()  # next couple of lines are used to place legend outside of the box
ax_motor12.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # shrinks box width by 80 % 
ax_motor12.legend(loc = 'center left', bbox_to_anchor=(1,0.5), ncol = 2) # loc in this case where the plot will be NOT the legend! bbox to anchor work with figure has dimensions 1x1x1 
plt.title('T/W vs RPM (PROPS - FILTERED(2))')
plt.ylabel("T/W [g/W]")
plt.xlabel("RPM [rev/s]") 
plt.xlim(10000,40000)
plt.ylim(1,8)
plt.show()     

#%% OLD CODE USES OG PROP DICTIONARY AND PRODUCES PLOTS
### RPM VS THRUST ####
'''
plt.figure("RPM vs Thrust")
for i in range(343,417):
    prop_df = PROP_dct[NO_DATtitles[i]]       # selects dataframe
    prop_title = NO_DATtitles[i]            # sets name - used for label
    plt.scatter(prop_df['Thrust'],prop_df['RPM'], label = prop_title)
plt.title('RPM vs Thrust at V_FREE = 23 m/s')
plt.xlabel("Thrust [LBS]")
plt.ylabel("RPM [-]") 
plt.legend(loc = 'lower right')
plt.xlim(2.0,6.0)
plt.show()
''' 
'''
#### RPM vs thrust vs PE 3d plot#####
fig3 = plt.figure("RPM vs Thrust vs Pe")
fig3.clf()
ax = Axes3D(fig3) # check if right fig number here!
for i in range(343,417):
    prop_df = PROP_dct[NO_DATtitles[i]]       # selects dataframe
    prop_title = NO_DATtitles[i]            # sets name - used for label
    ax.scatter(prop_df['Thrust'],prop_df['RPM'],prop_df['Pe'], label = prop_title)
ax.set_xlabel('Thrust')
ax.set_ylabel('RPM')
ax.set_zlabel('Pe')
#plt.legend(ncol= 4, loc = 'lower right')
plt.grid(True)
plt.show()
'''
#%%###### SETTING UP & FILTERING(1) MOTOR DICTIONARY FOR 100% THRUST SETTING ######################################

#rows_import = 256         # choose import howmany rows? => excel rows + 1
motor_data = pd.read_excel("Motor_database_miniquad.xlsx",header = 0)         #Import Excel File with all the data
#motor_df = pd.DataFrame(columns= ['Name', 'TESTED_KV', 'motor_weight(g)', 'Prop_Tested(100)', 'T_Max(g)','T_avg_max(g)', 'I_max(A)', 'I_avg_max(A)', 'Volt_avg_min', 'Volt_max','RPM_max', 'RPM_min', 'RPM_avg_max', 'Watts', 'T/W(g/W)'])
MOTOR_names = motor_data['Name']
kv = motor_data['TESTED_KV']
mw = motor_data['motor_weight(g)']
prop_t = motor_data['Prop_Tested(100)']
t_avg = motor_data['T_avg_max(g)']
I_avgmax = motor_data['I_avg_max(A)']
volt_avgmin = motor_data['Volt_avg_min']
rpm_avgmax = motor_data['RPM_avg_max'] 
m_pwr = motor_data['Watts']
m_tw = motor_data['T/W(g/W)']

# for-loop to split up excel into sections based on motor name + imported to motor dictionary
# MOTOR FILTER 1: BASED ON THRUST, T/W & RPM
# USES ALL prop types tested for every motor
MOTORFILT1_dct = {}
motor_name1 = MOTOR_names[0] # first motor name in excel, to give first value for if statement
m_start = 0 # must be zero!
motor_list_FILT1 = [] # list of motors passing filter 1
motor_list_og = []
facA = float(thrust_factor_avg)+thrust_factor_margin # uses the AVG dyn-static factor found from propeller database to look for motors with higher static thrust -> this should ultimately give motors with higher thrust
facB = float(thrust_factor_min)+thrust_factor_margin # uses the MIN dyn-static factor found from propeller database to look for motors with higher static thrust -> this should ultimately give motors with higher thrust (in general)
for j in range(0,len(MOTOR_names)):
   motor_df = pd.DataFrame(columns= ['TESTED_KV', 'motor_weight(g)', 'Prop_Tested(100)','T_avg_max(g)', 'I_avg_max(A)', 'Volt_avg_min', 'RPM_avg_max', 'Watts', 'T/W(g/W)'])
   for i in range(m_start,len(MOTOR_names)):
        motor_namei = MOTOR_names[i]
        if motor_namei in motorskiplist: # skips motors which are not possible inputs into ECALC
            pass
        elif motor_name1 == motor_namei: # checks whether were still working with the same motor
            if m_tw[i] > m_tw_bot and m_tw[i] < m_tw_top and rpm_avgmax[i] > m_rpm_bot and rpm_avgmax[i] < m_rpm_top and t_avg[i] > m_thrust_bot*facA and t_avg[i] < m_thrust_top*facA: # this is filter 1!
                motor_row = pd.DataFrame([[kv[i],mw[i],prop_t[i],t_avg[i],I_avgmax[i],volt_avgmin[i],rpm_avgmax[i],m_pwr[i],m_tw[i]]], columns= ['TESTED_KV', 'motor_weight(g)', 'Prop_Tested(100)','T_avg_max(g)', 'I_avg_max(A)', 'Volt_avg_min', 'RPM_avg_max', 'Watts', 'T/W(g/W)'])
                motor_df = pd.concat([motor_df,motor_row],ignore_index = True)
        elif motor_name1 != motor_namei and len(motor_df['TESTED_KV']) > 0: # negates empty dataframes from list
            MOTORFILT1_dct[motor_name1] = motor_df
            motor_list_FILT1.append(motor_name1)    # must be before name swith line
            motor_list_og.append(motor_name1)       # must be before name swith line
            motor_name1 = motor_namei               # must be after dictionary line
            m_start = i                             # ensures the loop does not start from the beginning again
            break
        else:
            motor_list_og.append(motor_name1)       # must be before name swith line
            motor_name1 = motor_namei               #
            m_start = i                             # ensures the loop does not start from the beginning again
            break
print('TOP thrust limit INcreased by factor: ', facA, '// BOTTOM DEcreased by factor:',facB)  
print('TOP thrust limit (MOTOR FILTER): ', m_thrust_top*facA, '  grams // BOTTOM thrust limit:',m_thrust_bot*facB, 'grams')  
print('OG motor list of',len(motor_list_og),'motors FILTERED(1) down to motor dictionary of',len(motor_list_FILT1),' motors')  
print('--------DONE WITH FILTERED(1) MOTOR DICTIONARY-------- ')
#%% PLOTTING FILTERED(1) MOTOR DATABASE (T/W vs RPM) for all propellers tested for every motor
'''
fig_motor1 = plt.figure('T/W vs RPM (MOTORS - FILTERED(1))')
fig_motor1.clf()
ax_motor1 = plt.subplot(111)
for i in range(0,len(motor_list_FILT1)):
    motor_title = motor_list_FILT1[i] # used for label and accessing dictionary
    motor_df =  MOTORFILT1_dct[motor_title]
    plt.scatter(motor_df['RPM_avg_max'],motor_df['T/W(g/W)'], label = (motor_title))
plt.title('T/W vs RPM (MOTORS - FILTERED(1))')
plt.ylabel("T/W [g/W]")
plt.xlabel("RPM [rev/s]") 
box = ax_motor1.get_position()  # next couple of lines are used to place legend outside of the box
ax_motor1.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # shrinks box width by 80 % 
ax_motor1.legend(loc = 'center left', bbox_to_anchor=(1,0.5), ncol = 2) # loc in this case where the plot will be NOT the legend! bbox to anchor work with figure has dimensions 1x1x1 
plt.xlim(10000,45000)
plt.ylim(1,8)
plt.show()  
'''
#%% MOTOR FILTER 2: CHOOSES ONLY THE MAX - MIN - AVG value of all propellers tested for every motor
# BUILD FILTER THAT ONLY SHOWS/PLOTS ONLY FOR THE MAX - MIN OR AVERAGE OF THE TESTED PROPELLERS?
MOTORFILT2_dct = {}
MOTORFILT2_keys = [] # stores keys of filt2 motor dictionary
MATCH_MOTOR_DF = pd.DataFrame(columns= ['TESTED_KV', 'motor_weight(g)', 'Prop_Tested(100)','T_avg_max(g)', 'I_avg_max(A)', 'Volt_avg_min', 'RPM_avg_max', 'Watts', 'T/W(g/W)','MOTOR_NAME'])
#MATCH_MOTOR_DF is used for MATCHING TOOL
propt5_3bladelist = ['HQ v1s 5x4x3','DAL Cyclone 5x4x3','Gemfan 5149 5.1x4.9x3','HQ v1s 5x4x4PC','HQ v1s 5x4.3x3PC','HQ 5x4x3GF','DAL Cyclone 5x4.5x3','T-Motor 5143 5.1x4.3x3','HQ v1s 5x4.5x3','DAL Cyclone 5x4.6x3','HQProp 5x4x3','DAL 5x4.5x3HBN','GemFan 5x4.5x3','DAL 5x4x4','HQ v1s 5x4x4PC','DAL 5x4.5x3BN','MyWing 5x4.5x3','HQProp 5x4x3GF','HQProp 5x4x3GF','Lumenier Buttercutter 5x5x3','HQ 5x4x6','Lumenier 5x5x3BC','Lumenier BC 5x5x3','HQ 5x4x4'] # list of 3blade-  5 inch tested propellers
propt5_4bladelist = ['HQ v1s 5x4x4PC','DAL 5x4x4','HQ v1s 5x4x4PC','HQ 5x4x6','HQ 5x4x4'] # list of 4 or more blade-  5 inch tested propellers
propt6_list = ['HQ 6x4.5','KingKong 6x4','HQ 6x3.5','HQ 6x4.5x3','HQ v1s 6x3x3PC','HQ v1s 6x4x3PC','Lumenier Buttercutter 6x4'] # list of all 6 inch tested propellers
for i in range(0,len(motor_list_FILT1)): # USES THE MAX T/W OF EVERY MOTOR & excludes certain prop types- A.K.A finds winning propeller
    motor_title2 = motor_list_FILT1[i]
    motor_df2 = MOTORFILT1_dct[motor_title2] 
    #tw2 = motor_df2['T/W(g/W)']
    propt2 = motor_df2['Prop_Tested(100)']
    motorFILT2i_df = pd.DataFrame(columns = motor_df2.columns) # new dataframe for every key - intermediary product used to check for max
    motorFILT2_df = pd.DataFrame(columns = motor_df2.columns) # new dataframe for every key - end product
    for j in range(0,len(propt2)): # this for loop will check for 5 or 6 inch propellers
        #motor_row22j = 0 # ensures motor row is reset
        if propt2[j] not in propt6_list:  # use if only use 5 inch props
            if propt2[j] not in propt5_4bladelist:  # excludes 4 or more blade 5 inch props
        #if propt2[j] in propt6_list: # use if want to use only 6 inch props
                motor_row22j = motor_df2[j:j+1]  
        try:
            motorFILT2i_df = pd.concat([motorFILT2i_df, motor_row22j],ignore_index = True) # creates dataframes of multiple tested props only 5 or 6 inch
        except(TypeError):
            pass
    try:
        tw2i = motorFILT2i_df['T/W(g/W)']            #needed to find in new 5 or 6 inch df the propt with highest value for t/w
        indexmax = tw2i.idxmax()                     # finds max of the t/w row of every motor
        motor_row22 = motorFILT2i_df[indexmax:indexmax+1] # uses row of said t/w max and uses it for new dict
        motor_row22_match = motorFILT2i_df[indexmax:indexmax+1].copy()
        #indexmax = tw2.idxmax()                     # finds max of the t/w row of every motor
        #motor_row22 = motor_df2[indexmax:indexmax+1] # uses row of said t/w max and uses it for new dict
        #motor_row22_match = motor_df2[indexmax:indexmax+1].copy()
        motor_row22_match['MOTOR_NAME'] = motor_title2 # adds another column to dataframe with motor name 
        motorFILT2_df = pd.concat([motorFILT2_df, motor_row22],ignore_index = True)
        MOTORFILT2_dct[motor_title2] = motorFILT2_df
        MOTORFILT2_keys.append(motor_title2)
        MATCH_MOTOR_DF = pd.concat([MATCH_MOTOR_DF,motor_row22_match],ignore_index = True)
    except(TypeError):
        pass
#print('OG motor list of',len(motor_list_og),'motors FILTERED(1) down to motor dictionary of',len(motor_list_FILT1),' motors')  
print('WINNING propellers chosen for every motor + 5 or 6 inch setting applied!')
print('FILT1 MOTOR DICT of',len(motor_list_FILT1),'motors filtered down to dictionary FILT2 of ',len(MOTORFILT2_keys), 'motors' )
print('--------DONE WITH FILTERED(2) MOTOR DICTIONARY-------- ')
#%% PLOTTING FILTERED(2) MOTOR DATABASE (T/W vs RPM) for WINNING propellers tested for every motor

fig_motor2 = plt.figure('T/W vs RPM (MOTORS - FILTERED(2))')
fig_motor2.clf()
ax_motor2 = plt.subplot(111)
for i in range(0,len(MOTORFILT2_keys)):
    motor_title = MOTORFILT2_keys[i] # used for label and accessing dictionary
    motor_df =  MOTORFILT2_dct[motor_title]
    plt.scatter(motor_df['RPM_avg_max'],motor_df['T/W(g/W)'], label = (motor_title))
plt.title('T/W vs RPM (MOTORS - FILTERED(2))')
plt.ylabel("T/W [g/W]")
plt.xlabel("RPM [rev/s]") 
box = ax_motor2.get_position()  # next couple of lines are used to place legend outside of the box
ax_motor2.set_position([box.x0, box.y0, box.width * 0.8, box.height]) # shrinks box width by 80 % 
ax_motor2.legend(loc = 'center left', bbox_to_anchor=(1,0.5), ncol = 1) # loc in this case where the plot will be NOT the legend! bbox to anchor work with figure has dimensions 1x1x1 
plt.xlim(10000,40000)
plt.ylim(1,8)
plt.show()  


#%%######## MATCHING TOOL  ##########################

### PROP & MOTOR PLOT TOGETHER ####
# MATCH_MOTOR_DF - filtered(1&2) dataframe for motors
# propMATCH_df - filtered(1&2) dataframe for propellers
fig_MATCH = plt.figure('T/W vs RPM (MATCHING)')
fig_MATCH.clf()
plt.scatter(MATCH_MOTOR_DF['RPM_avg_max'],MATCH_MOTOR_DF['T/W(g/W)'],label = 'Motors')
plt.scatter(propMATCH_df['RPM[rev/s]'],propMATCH_df['T/W[g/W]'],label = 'Propellers')
plt.title('T/W vs RPM (MATCHING- FILTER1&2 APPLIED)')
plt.ylabel("T/W [g/W]")
plt.xlabel("RPM [rev/s]")
plt.legend(loc = 'lower right') 
plt.xlim(10000,40000)
plt.ylim(1,8)
plt.show()  
#%%### MATCHING: FINDS FOR EVERY PROPELLER THE CLOSEST PROPELLER - ONLY WORKS IF THAT DISTANCE IS SHORTER THAN minimum distance ####
# FOR MATCHING TO be valid: propeller has to be under the motor wrt T/PWR since that means the motor will always be able to supply
# -> the power required by the propeller, thus the motor will not have to work harder than it is able to (does not cause burnout!)
# motor rpm may be both to the left and right of prop (more margin allowed here)! 
#Minimum_distance =  0.12 # absolute minimum distance between two points FOR MATCHING!
MATCH_keys_df = pd.DataFrame(columns = ['Distance','Motor','Prop']) # dataframe to store all keys
DUOS_main = {}  # stores duos of main dictionary
for i in range(0,len(propMATCH_df['RPM[rev/s]'])):
    DIS1 = Minimum_MAXdistance # starting value - resets
    xprop = float(propMATCH_df['RPM[rev/s]'][i])/10000  # has to be scaled in order to make x & y axis equally impactful
    yprop = float(propMATCH_df['T/W[g/W]'][i])
    DUOS_i = {}  # DUO MATCH STORAGE
    motor_name = 'na'  # starting name - resets
    motor12_df = pd.DataFrame(columns = MATCH_MOTOR_DF.columns)
    for j in range(0,len(MATCH_MOTOR_DF['T/W(g/W)'])):
        xmotor = float(MATCH_MOTOR_DF['RPM_avg_max'][j])/10000 # has to be scaled in order to make x & y axis equally impactful
        ymotor = float(MATCH_MOTOR_DF['T/W(g/W)'][j])
        yDIS = yprop - ymotor   # T/PWR distance between motor and prop
        xDIS = xprop - xmotor   # RPM distance between motor and prop
        if yDIS < 0.000000 and xDIS < 0.000000: # checks if motor is above & to right of propeller  - thus if yDIS is negative  - switch off for 6 inch
            #DIS = np.sqrt((yDIS)**2 + (xDIS)**2) # finds distance between points 
            if abs(yDIS) > DIS1 and abs(xDIS) < rpm_margin/10000: # # sets max t/pwr distance(atm)
                    motor_row = MATCH_MOTOR_DF[j:j+1]  # this will be final variable if it is not replaced
                    motor_name = MATCH_MOTOR_DF['MOTOR_NAME'][j] # this will be final variable if it is not replaced
                    DIS1 = abs(yDIS) # this will be final variable if it is not replaced
        else: 
            pass
    if DIS1 != Minimum_MAXdistance: # negates empty dataframes and points for which minimum distance cannot be found under minimum!
        DUOS_mainName = 'Distance {}'.format(round(DIS1,5))
        prop_name = propMATCH_df['NAME'][i]
        key_row = pd.DataFrame([[DUOS_mainName, motor_name, prop_name]], columns = ['Distance','Motor','Prop'])
        MATCH_keys_df = pd.concat([MATCH_keys_df, key_row], ignore_index = True)
        DUOS_i[propMATCH_df['NAME'][i]] = propMATCH_df[i:i+1]
        DUOS_i[motor_name] = pd.concat([motor12_df, motor_row], ignore_index = True)
        DUOS_main[DUOS_mainName] = DUOS_i
MATCH_keys_df= MATCH_keys_df.sort_values(['Distance'], ascending = False) # sorts dataframes based on best match! - important for verification
print('--------- FOUND AND STORED MOTOR & PROP DUOS -----------')   
#%%######## Battery database -> dictionary for different battery S-types #############
battery_data = pd.read_excel("BATTERY_DATABASE.xlsx",header = 0)         #Import Excel File with all the data
S_type = battery_data['S_type']
BATLIST = ['BAT_4S','BAT_5S','BAT_6S']
start_b = 0 # starting value
BAT_dct= {}
for j in range(0,len(BATLIST)):
        bat_name = BATLIST[j] 
        battery_df = pd.DataFrame(columns = battery_data.columns)
        for i in range(start_b,len(S_type)):
            if S_type[i] !=  'S_type': # starts new dataframe when new S class is discovered        
                bat_row = battery_data[i:i+1]
                battery_df = pd.concat([battery_df,bat_row], ignore_index = True)
            else:
                start_b = i+1
                break
        BAT_dct[bat_name] = battery_df
print('--------- CREATED BATTERY DATAFRAMES -----------')
#%%####### MATCH BATTERY TO PROP & MOTOR DUO (includes one filter for battery)   #############
# battery filter checks current & capacity requirement, minimum c-rate (not really important) NOT max battery weight
C_AVG = [65.571,61.364,65.922]  #average C_rate for 4,5 & 6S - obtained MANUALLY!
bot4S = 12.9500
bot5S = 16.6500
bot6S = 20.3500
#top6S = 22.2000
top6S = 25.000
for i in range(0,len(MATCH_keys_df['Distance'])):
    duos_key = MATCH_keys_df['Distance'][i] # obtains key to acces certain DUOS dictionary
    motor_key = MATCH_keys_df['Motor'][i]  # obtains key to acces motor in certain DUOS dictionary
    #prop_key = MATCH_keys_df['Prop'] # obtains key to acces propeller in certain DUOS dictionary
    bat_dicti = DUOS_main[duos_key]
    #propi_df = bat_dicti[PROP_MATCH_keys[i]] # should be choose propeller instead?
    motori_df = bat_dicti[motor_key] 
    m_volt = float(motori_df['Volt_avg_min'][0])
    m_cur = float(motori_df['I_avg_max(A)'][0])
    current_thres = m_cur*n_motors+I_dry # sets current threshold required by entire drone for battery
    capa_thres = (current_thres)/3600*FT*1000/eta_bat+(I_dry)/3600*FT_idle/eta_bat     # battery capacity mAh - threshold (second term = idle flight time)
    #FT_eff = capa_thres*eta_bat*3600/(1000*current_thres) # effective flight time REQUIRED at 100% - thus idle time is short time at 100%
    if m_volt > bot4S and m_volt < bot5S: # this if statement ensure only the same S-type batteries (as motor) are examined 
        s_key = BATLIST[0]    # defines key needed to access battery dictionary
        c_avgi = C_AVG[0]        # defines c-rate average needed to downscale give c-rate
    elif m_volt > bot5S and m_volt < bot6S:
        s_key = BATLIST[1] 
        c_avgi = C_AVG[1]
    elif m_volt > bot6S and m_volt < top6S:
        s_key =  BATLIST[2] 
        c_avgi = C_AVG[2]
    else:
        print('CANT GIVE S-TYPE LABEL')
        print(duos_key)
        break
    batteries = BAT_dct[s_key]   # Uses s_key to access battery dictionary for correct motor S type
    battery_df = pd.DataFrame(columns = battery_data.columns) # empty dataframe for winning battery 
    for j in range(0,len(batteries)):
        b_capi = float(batteries['Capacity [mAh]'][j])
        b_weighti = float(batteries['Weight [g]'][j])
        b_c_ratei = (float(batteries['C-rating'][j])-(abs(float(batteries['C-rating'][j])-c_avgi))*c_rate_per) # c-rate is DOWN-scaled by [diff*percentage] to account for unreliable advertising
        b_curi = b_c_ratei*(b_capi/1000)  # current is calculated with using adapted c-rate and capacity
        if b_curi > current_thres*on_off1 and b_capi > capa_thres*on_off1 and b_weighti < max_Bweight and b_c_ratei < c_rate_max*on_off1: # filter 1 battery
            bat_row = batteries[j:j+1]
            battery_df = pd.concat([battery_df,bat_row], ignore_index = True)
    indexmin = battery_df[['Weight [g]']].values.argmin()  # finds row index of the battery with lowest weight, which did pass the filter
    #print(indexmin)
    battery_win = battery_df[indexmin:indexmin+1]
    bat_dicti['Battery'] = battery_win.reset_index(drop = True) # adds battery to prop & motor duo
    #bat_dicti['FT@max[sec]'] = FT_eff # adds effective flight time required! to the configuration - the battery will be able to fly for longer
print('------- ADDED WINNING BATTERY TO MOTOR & PROP DUOS ---------')