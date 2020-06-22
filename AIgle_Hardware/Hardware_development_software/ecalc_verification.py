import os
import numpy as np
import pandas as pd
import re
from io import StringIO
from IPython.display import display
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# obtaining paramters from parse file - 6 inch
'''
from parse_dat_files import DUOS_main
from parse_dat_files import MATCH_keys_df
from parse_dat_files import mass_estimate
from parse_dat_files import T_W_REQ
from parse_dat_files import T_W_toplim
from parse_dat_files import eta_motor
from parse_dat_files import eta_bat
from parse_dat_files import FT
from parse_dat_files import FT_idle
from parse_dat_files import I_dry
from parse_dat_files import v_free
from parse_dat_files import max_tilt
from parse_dat_files import n_motors
'''
# for 5 inch
from parse_dat_files5in import DUOS_main
from parse_dat_files5in import MATCH_keys_df
from parse_dat_files5in import mass_estimate
from parse_dat_files5in import T_W_REQ
from parse_dat_files5in import T_W_toplim
from parse_dat_files5in import eta_motor
from parse_dat_files5in import eta_bat
from parse_dat_files5in import FT
from parse_dat_files5in import FT_idle
from parse_dat_files5in import I_dry
from parse_dat_files5in import v_free
from parse_dat_files5in import max_tilt
from parse_dat_files5in import n_motors



#scikit for regression
# ENSURE THAT THE VERIFICATION FILE HAS THE RIGHT ORDER (BEST TO WORST MATCH) FROM LEFT TO RIGHT IN THE EXCEL FILE!
#%%##### PARAMETERS  #################
    # notes
# incorporate flight time idle as seconds of full!
# should be give an estimate of eta_full? e.g. 60% can we reason this?
#FT_eff = capa_thres*eta_bat*3600/(1000*current_thres)
# thrust to weight uses bottom limit(REQ!) -> IS THIS CORRECT
#%%#### OBTAINING ECALC VERIFICATION DATAFRAMES #############
veri_data = pd.read_excel("ECALC_test5in.xlsx", header = 3, nrows= 93)         # Import Excel File with all the data
clist = (list(veri_data))[2:] # list of configurations
n_configs = len(clist) # amount of configurations
clist.insert(0,'Property')
clist.insert(1,'unit')
complist = ['BATTERY','CONTROLLER','Accessories','MOTOR@max','MOTOR@hover','DRIVE','MULTICOP'] # list of components - seperate dataframe will be made for these
skiplist= ['oz','W/lb','mph','mi','ft/min','in²','°F','oz/W','@'] # list of all things that should be skipped when found
check =  veri_data[clist[2]] # defines column used to check database
start_c = 1 # start at 1 because 0 is a row of nans
comp_dct = {}
for j in range(0,len(complist)):
    vdata_main = pd.DataFrame(columns = veri_data.columns)
    for i in range(start_c,len(check)):
        if pd.isna(check[i]) and pd.isna(check[i+1]) and i != 47: # double nan - start new dataframe - irregular gap at index 47!
            start_c = i+3 # makes sure loop is not started at beginning again
            break
        elif veri_data['Unnamed: 1'][i] in skiplist: # gets rid of rows unit in skip list
            pass
        else:
            veri_row = veri_data[i:i+1]
            vdata_main = pd.concat([vdata_main,veri_row],ignore_index = True)      
    vdata_main.columns = clist # renames columns 
    comp_dct[complist[j]] = vdata_main
print('------- DONE WITH ECALC DATAFRAMES---------')
#%%## COMPARING VERIFICATION DATA TO OWN TRIPLETS! ###
# comparison occurs by comparing verification data with own in (new-old)/old -> old = veri data!
#DUOS_main = own data results
#MATCH_keys_df = keys req to acces duos_main
### ABSOLUTE & RELATIVE ERROR DEFINITIONS ### 
def error_r(a,b): # relative error
    error_r = (float(a)-float(b))/float(b)*100 # in percentage
    return(error_r)
def error_a(a,b): # absolute error
    error_a = float(a)-float(b)
    return(error_a)

### COMPARING MOTOR AT 100% ###
verimotor_df = comp_dct['MOTOR@max']  # motor at 100%
veribat_df = comp_dct['BATTERY']  # battery related dataframe
vericonfig_df = comp_dct['MULTICOP']  # general config related dataframe - last part of excel
veridrive_df = comp_dct['DRIVE']  # drive related dataframe
mveri_results_2c = [['m Weight[g]','abs'],['m Current[Amp]','%'],['m Voltage[V]','%'],['m RPM[rev/s]','%'],['m Input power[W]','%'],['m Efficiency[%]','abs']] # first two columns of motor results df
mveri_results_df = pd.DataFrame(mveri_results_2c, columns = ['Property','rel or abs?']) # dataframe for motor results 
start_r = 0 # defines starting point - ensures duplictation will not occur
### COMPARING GENERAL CONFIGURATION PROPERTIES ###
#totveri_results_2c = [['Total Weight[g]','%'],['T/W','%'],['Overall efficiency','%'],['Max tilt angle','%'],['Max speed','%'],['Flight time @max','abs']] # first two columns of results df
totveri_results_2c = [['Total Weight[g]','%'],['T/W','abs'],['Max tilt angle [°]','abs'],['Max speed','%'],['Flight_time@max[sec]','abs']] # first two columns of results df
totveri_results_df = pd.DataFrame(totveri_results_2c, columns = ['Property','rel or abs?']) # dataframe for general config results 

for i in range(0,len(MATCH_keys_df['Motor'])):
    owntrio = DUOS_main[MATCH_keys_df['Distance'][i]]
    ownmotor_df = owntrio[MATCH_keys_df['Motor'][i]]
    ownbat_df = owntrio['Battery']
    for j in range(start_r,n_configs):
        config = clist[2+j] # skips property and unit column
        # motor related
        cmweight = error_a(ownmotor_df['motor_weight(g)'][0],verimotor_df[config][1])# absolute difference in motor weight [g]
        ccurrent = error_r(ownmotor_df['I_avg_max(A)'][0],verimotor_df[config][2]) # relative difference in current [%]
        cvolt = error_r(ownmotor_df['Volt_avg_min'][0],verimotor_df[config][3]) # relative difference in volt [%]
        crpm = error_r(ownmotor_df['RPM_avg_max'][0],verimotor_df[config][4]) # relative difference in rpm [%]
        cpwr = error_r(ownmotor_df['Watts'][0],verimotor_df[config][5]) # relative difference in input power [%]
        ceta = error_a((eta_motor*100),verimotor_df[config][7]) # absolute difference in efficiency [%]
        mveri_results_col = pd.DataFrame([[cmweight],[ccurrent],[cvolt],[crpm],[cpwr],[ceta]], columns = [config]) # to be appended column for motor results df
        # general drone config related
        mFTeff = float(ownbat_df['Capacity [mAh]'][0])*eta_bat*3.600/(float(ownmotor_df['I_avg_max(A)'])*n_motors + I_dry) # effective flight time at 100% of ownmotor
        cFT = error_a(mFTeff,float(veribat_df[config][9])*60) # absolute error for flight time
        ctweight = error_r(mass_estimate,vericonfig_df[config][1]) # relative error for total weight
        ctTW = error_a(T_W_REQ,veridrive_df[config][2]) # absolute error for thrust-to-weight
        ctTilt = error_a(max_tilt,vericonfig_df[config][3]) # absolute error tilt angle
        ctspeed = error_r(v_free,float(vericonfig_df[config][4])/3.6) # relative error speed
        tveri_results_col = pd.DataFrame([[ctweight],[ctTW],[ctTilt],[ctspeed],[cFT]], columns = [config]) # to be appended column for total results df
        start_r = j+1
        break
    totveri_results_df = pd.concat([totveri_results_df,tveri_results_col], axis = 1) # appends general configuration results
    mveri_results_df = pd.concat([mveri_results_df,mveri_results_col], axis = 1) # appends motor results
VERI_results_dct = {} # dictionary to store all verification results
VERI_results_dct['Motor'] = mveri_results_df 
VERI_results_dct['Config'] = totveri_results_df
print('--- Obtained error for general config & motor related properties -----')
print('--- COMPLETELY DONE WITH VERIFICATION SCRIPT -----')
#%%# GENERAL DRONE PROPERTIES RESULTS

#n_configs
#motor_comp_df = pd.Dataframe(columns = verimotor_df.columns)

