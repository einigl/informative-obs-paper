import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(2, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import helpers
from pdr_util import is_hyperfine

### SETTINGS ###

emir_bands = ["E090", "E150", "E230", "E330"]

mols_to_remove = ["so"]
lines_to_remove = []
keep_hyperfines = False

observing_time = helpers.orionb_obstime # By pixel, in minutes (before: 0.75)
snr_percentile = 99
snr_threshold = 3

n_channels = 20
dv = 0.5

csvname = "emir_table_filtered"

################

# Data loading
df_full = pd.read_csv("../ref_tables/full_table.csv")
df_emir = pd.read_csv("../ref_tables/emir_table_noise.csv")

init_nb = len(df_emir)

# Dataframes merging
df_emir = df_emir.merge(df_full[["line_id", "molecule", f"per{snr_percentile}_intensity"]], on='line_id', how='inner')

print(
    "LINES OBSERVABLE BY EMIR\n\n", df_emir['EMIR band'].value_counts(), end="\n\n"
)

# Keep selected EMIR bands
df_emir = df_emir[df_emir["EMIR band"].isin(emir_bands)]

# Display number of lines by EMIR bands
print(
    f"LINES BELONGING TO BANDS {emir_bands}\n\n", df_emir['EMIR band'].value_counts(), end="\n\n"
)

# Removing of unwanted molecules
before = len(df_emir)
df_emir = df_emir[~df_emir["molecule"].isin(mols_to_remove)]

# Removing of unwanted lines
df_emir = df_emir[~df_emir["line_id"].isin(lines_to_remove)]
after = len(df_emir)

print(
    f"AFTER REMOVAL OF UNWANTED SPECIES ({before - after})\n\n", df_emir['EMIR band'].value_counts(), end="\n\n"
)

# Remove hyperfine lines except the brightest one
if not keep_hyperfines:
    lines = df_emir.sort_values(by=[f"per{snr_percentile}_intensity"], ascending=False)["line_id"].to_list()
    to_remove = []

    k = 0
    count = 0
    while k < len(lines):
        line = lines[k]
        if not is_hyperfine(line):
            k += 1
            continue
        i = k+1
        while i < len(lines):
            if is_hyperfine(line, lines[i]) :
                to_remove.append(lines[i])
                lines.pop(i)
                count += 1
            else:
                i += 1
        k += 1

    df_emir = df_emir[df_emir["line_id"].isin(lines)]

    print(
        f"AFTER REMOVAL OF UNWANTED HYPERFINE LINES ({count})\n\n", df_emir['EMIR band'].value_counts(), end="\n\n"
    )

# Remove lines whose SNR percentile is not high enough
    
def convert_unit(T, nu, dv):
    """
    [T]: K
    [nu]: Hz
    [dv]: km.s^-1
    """
    kb = 1.380_649e-23 # m^2.kg.s^-2.K^-1
    c = 299_792_458 # m.s^-1
    return (2 * 1e6 * (nu/c)**3 * kb) * T * dv

def integrate_rms(rms, n):
    """
    rms: RMS value for a single velocity channel
    N: number of velocity channel to integrate
    """
    return n**0.5 * rms

freqs = df_emir["freq"].to_numpy()
intensities = df_emir[f"per{snr_percentile}_intensity"].to_numpy()
rms = df_emir["Noise RMS (K) [1 min]"].values

# Remove column of reference noise level
df_emir = df_emir.drop(columns="Noise RMS (K) [1 min]")

# Store reference (1 min obs. time) integrated noise RMS
rms = integrate_rms(rms, n_channels)
df_emir.insert(
    df_emir.shape[1], f"Noise RMS (K) [1 min]",
    rms
)
df_emir.insert(
    df_emir.shape[1], f"Noise RMS (Mathis units, log10) [1 min]",
    np.log10(convert_unit(rms, 1e9*freqs, dv))
)

# Store integrated noise RMS for effective observing time
rms = rms / np.sqrt(observing_time)
rms_unit = convert_unit(rms, 1e9*freqs, dv)
rms_unit_log = np.log10(rms_unit)

snr = 10**(intensities - rms_unit_log)
snr_db = 10 * np.log10(snr)

df_emir.insert(df_emir.shape[1], f"Noise RMS (K) [{round(observing_time, 2)} min]", rms)
df_emir.insert(df_emir.shape[1], f"Noise RMS (Mathis units, log10) [{round(observing_time, 2)} min]", rms_unit_log)
df_emir.insert(df_emir.shape[1], f"SNR (per{snr_percentile})", snr)
df_emir.insert(df_emir.shape[1], f"SNR (per{snr_percentile}, dB)", snr_db)

df_emir = df_emir.sort_values(by=[f'SNR (per{snr_percentile}, dB)'], ascending=False)
df_emir = df_emir[df_emir[f"SNR (per{snr_percentile})"] >= snr_threshold]

print(
    "AFTER FILTERING SNRS\n\n", df_emir['EMIR band'].value_counts(), end="\n\n"
)

end_nb = len(df_emir)
print(f"{init_nb} -> {end_nb} lines")

# Sort by ascending frequency
df_emir = df_emir.sort_values(by=[f'freq'], ascending=True)

# Save new DataFrame
df_emir.to_csv(f"{csvname}.csv", index=False)
