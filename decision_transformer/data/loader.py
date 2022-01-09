import pickle
import os
import numpy as np


def load_pickle(name):
    infile = open(name,'rb')
    new_dict = pickle.load(infile)

    print(len(new_dict))
    print(new_dict[0].keys())

    print("Observation shape", new_dict[0]["observations"].shape)
    print("Action shape", new_dict[0]["actions"].shape)
    print("Reward shape", new_dict[0]["rewards"].shape)
    print("Terminal shape", new_dict[0]["terminals"].shape)

    for i in range(20):
        print("Observation shape", new_dict[i]["observations"].shape)
    print("Observation shape", new_dict[-1]["observations"].shape)
    infile.close()

    return new_dict
    

def save_pickle(name):
    sample_data = [{"key1": "value1", "key2": "value2"}]

    filename = str(name)+".pkl"
    outfile = open(filename, "wb")

    pickle.dump(sample_data, outfile)
    outfile.close()


if __name__ == "__main__":

    env_name = "drone"
    env_diff = "expert"

    dataset = load_pickle(name=env_name + "-" + env_diff + "-v2.pkl")

    # save_pickle("sample")
