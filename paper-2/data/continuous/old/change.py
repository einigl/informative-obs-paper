import os
import pickle

from tqdm import tqdm

files = os.listdir()
files = [f for f in files if f.endswith(".pickle")]

for file in tqdm(files):
    with open(file, "rb") as f:
        data = pickle.load(f)

    for i in range(len(data)):
        entry = data[i].pop("windows")
        data[i]["features"] = entry

    with open(file, "wb") as f:
        pickle.dump(data, f)
