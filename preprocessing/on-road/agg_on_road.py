import pandas as pd
import xarray as xr

# Set wide display options for debugging
pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10000)

# base1 = xr.open_dataset('FATE_gridded_Baseline.nc')
base2 = pd.read_csv('/Users/j.benoit/Documents/Roadmap/Global Health Study/GHS Results July 24/Concatenated_GHS_Results/results_summary_with_wtt_July_24_Global.csv')
base3 = pd.read_csv('results_summary_wtt_by_sector.csv')

upstream = pd.read_csv('upstream_emissions.csv')
fuel_cols = ['Gasoline', 'Ethanol', 'Diesel', 'CNG', 'LNG']
# Melt the upstream emissions data
upstream = upstream.melt(id_vars=['spec'], value_vars=fuel_cols, var_name='Fuel', value_name='g/MJ')
# Convert from g/MJ to g/PJ
upstream['g/MJ'] = upstream['g/MJ']*1e6

# base1 = base1.to_dataframe().reset_index()
# base1 = base1.groupby(['year', 'spec', 'sect']).mean().reset_index().drop(columns=['lat', 'lon'])
# base1['pm']=base1['pm']*5.1e14*1e-6

base2 = base2[base2['Scenario'].isin(['All Measures', 'Baseline'])]

fuels = base2[['Scenario', 'Fuel', 'CY', 'PJ']].copy()
fuels = fuels.groupby(['Scenario', 'Fuel', 'CY']).sum(numeric_only=True).reset_index()

fuels = fuels.merge(upstream,how='outer', on='Fuel')


base2 = base2[['Scenario', 'CY', 'TTW CO2', 'WTT CO2', 'WTW CO2', 'TTW NOx', 'WTT NOx', 'WTW NOx', 'TTW BC', 'TTW OC']]
base2 = base2.groupby(['Scenario', 'CY']).sum().reset_index()

base3 = base3[base3['Scenario'].isin(['All Measures', 'Baseline'])]
base3 = base3.groupby(['Scenario', 'CY', 'Sector']).sum(numeric_only=True).reset_index()

# Use base 3 for all except CO2. Use base2 for CO2
base2 = base2[['Scenario', 'CY', 'TTW CO2', 'WTT CO2']]

# base3 = base3.pivot_table(index=['Scenario', 'CY', 'Sector'], columns='spec', values='pm').reset_index()
base3_ttw = base3[base3['Sector'].isin(['TRA_GSL','TRA_MD'])].groupby(['Scenario', 'CY']).sum(numeric_only=True).reset_index()
base3_ttw = base3_ttw[['Scenario', 'CY', 'BC']]

all = base2.merge(base3_ttw, on=['Scenario', 'CY'])
all.to_csv('Combined_CO2_BC.csv',index=False)

print()