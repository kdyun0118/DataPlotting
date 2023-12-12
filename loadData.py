from collections import defaultdict
import os
import numpy as np
import pickle
import wandb

def load_wandb_data(wandb_id,path, run_name):
    # read data
    api = wandb.PublicApi()
    run = api.run(wandb_id)
    history = run.scan_history()
    logs = []
    for row in history:
        logs.append(row)
    # print(logs) # list of dictionaries
    # print(logs[0]) # dictionaries
    # print(logs[0]['Rewards/pitchVel']) # a value corresponding to a key
    # save data
    isExist = os.path.exists(path)
    print("path exist? ",isExist)
    if not isExist:
        os.makedirs(path)
        print("path created ",path)
    pickle.dump(logs, open(path+"/"+run_name+".pkl", "wb"))

    print("Loading Wandb data completed!")

def load_data(path,run_name, data_types, data_length=20000, filter_period=100):
    logs = pickle.load(open(path+"/"+run_name+".pkl", "rb")) #list of dicts
    # print(len(logs))
    # print(list(logs[0].keys()))

    assert len(logs) >= data_length, "Too short data length." + str(len(logs))
    logs = logs[:data_length]

    filter_logs = defaultdict(list)
    for idx, log in enumerate(logs):
        if idx % filter_period == 0:
            noNan = True
            for data_type in data_types:
                try:
                    if (np.isnan(log[data_type])):
                        noNan = False
                except:
                    noNan = False

            if noNan:
                for data_type in data_types:
                    filter_logs[data_type].append(log[data_type])

    final_logs = dict()
    for data_type in data_types:
        final_logs[data_type] = np.array(filter_logs[data_type], dtype=np.float32)
    return final_logs

if __name__ == "__main__":
    os.environ["WANDB_API_KEY"] = "5804d2713f53ff0c9b549cd9ed8dd3614652c2c2"
    # taskname = "yawspin"
    # taskname = "sideflip"
    # taskname = "backflip"
    # taskname = "diagonalflip"
    # taskname = "consecutive_backflip"

    taskname = "curriculum ablation"

    # trainingMode = "SRB pretraining"
    # trainingMode = "SRB pretrained curriculum"
    # trainingMode = "SRB pretrained full"
    # trainingMode = "vanilla full"

    trainingMode = "direct transfer"
    # trainingMode = "100 iteration"
    # trainingMode = "200 iteration"
    # trainingMode = "300 iteration"
    # trainingMode = "400 iteration"
    # trainingMode = "1000 iteration"



    savePath = "/home/kdyun/Desktop/research record/Data/"+taskname+"/"+trainingMode

    wandb_id = "kdyun/curriculum ablation2/2pzmfsr1"
    run_name = "2023-08-23-23-49-45"

    load_wandb_data(wandb_id,savePath, run_name) # 앞의 wandb run을 뒤의 pickle로

    # data_types = ["Rewards/pitchVel","Validation/return"]
    # filter_logs = load_data(path=savePath,run_name="hello", data_types=data_types, data_length=32, filter_period=2)
    # print(filter_logs)