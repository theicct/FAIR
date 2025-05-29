import numpy as np
import matplotlib.pyplot as pl
import pandas as pd

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise
from fair.earth_params import seconds_per_year

f = FAIR(ch4_method='thornhill2021')

f.define_time(1750, 2100, 1)

# Define SSP scenarios
scenarios = ['ssp119']
f.define_scenarios(scenarios)


df = pd.read_csv("../tests/test_data/4xCO2_cummins_ebm3.csv")
models = df['model'].unique()
configs = []

for imodel, model in enumerate(models):
    for run in df.loc[df['model']==model, 'run']:
        configs.append(f"{model}_{run}")
f.define_configs(configs)

species, properties = read_properties()
#species = list(properties.keys())

f.define_species(species, properties)

f.allocate()

f.fill_species_configs()
fill(f.species_configs['unperturbed_lifetime'], 10.8537568, specie='CH4')
fill(f.species_configs['baseline_emissions'], 19.01978312, specie='CH4')
fill(f.species_configs['baseline_emissions'], 0.08602230754, specie='N2O')

df_volcanic = pd.read_csv('../tests/test_data/volcanic_ERF_monthly_175001-201912.csv', index_col='year')
df_volcanic[1750:].head()

f.fill_from_rcmip()

# overwrite volcanic
volcanic_forcing = np.zeros(351)
volcanic_forcing[:271] = df_volcanic[1749:].groupby(np.ceil(df_volcanic[1749:].index) // 1).mean().squeeze().values
fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")  # sometimes need to expand the array

initialise(f.concentration, f.species_configs['baseline_concentration'])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

df = pd.read_csv("../tests/test_data/4xCO2_cummins_ebm3.csv")
models = df['model'].unique()

seed = 1355763

for config in configs:
    model, run = config.split('_')
    condition = (df['model'] == model) & (df['run'] == run)
    fill(f.climate_configs['ocean_heat_capacity'], df.loc[condition, 'C1':'C3'].values.squeeze(), config=config)
    fill(f.climate_configs['ocean_heat_transfer'], df.loc[condition, 'kappa1':'kappa3'].values.squeeze(), config=config)
    fill(f.climate_configs['deep_ocean_efficacy'], df.loc[condition, 'epsilon'].values[0], config=config)
    fill(f.climate_configs['gamma_autocorrelation'], df.loc[condition, 'gamma'].values[0], config=config)
    fill(f.climate_configs['sigma_eta'], df.loc[condition, 'sigma_eta'].values[0], config=config)
    fill(f.climate_configs['sigma_xi'], df.loc[condition, 'sigma_xi'].values[0], config=config)
    fill(f.climate_configs['stochastic_run'], True, config=config)
    fill(f.climate_configs['use_seed'], True, config=config)
    fill(f.climate_configs['seed'], seed, config=config)

    seed = seed + 399

f.run()

t = f.temperature.loc[dict(scenario='ssp119', layer=0)].to_dataframe(name='temp').reset_index()
t = t.drop(columns=['scenario', 'layer'])
mean = t.groupby(['timebounds']).mean(numeric_only=True).reset_index()
upper_95 = t.groupby('timebounds').quantile(0.95, numeric_only=True).reset_index()
lower_95 = t.groupby('timebounds').quantile(0.05, numeric_only=True).reset_index()


# upper_95 = t.groupby('timebounds').rank(pct=True).reset_index()
fig, ax = pl.subplots()
ax.plot(mean['timebounds'], mean['temp'], label='mean');
ax.plot(upper_95['timebounds'], upper_95['temp'], label='upper 95th percentile');
ax.plot(lower_95['timebounds'], lower_95['temp'], label='lower 95th percentile');
pl.title('ssp119: temperature')
pl.xlabel('year')
pl.ylabel('Temperature anomaly (K)')
pl.ylim(-1,3)
ax.set_yticks(np.arange(-1, 3.1, 0.5))
ax.grid(alpha=0.5, axis='y')
ax.legend()
pl.show()