import matplotlib
import matplotlib.pyplot as plt

import pickle
import numpy as np
import loadData
import os
import copy

os.environ["WANDB_API_KEY"] = "5804d2713f53ff0c9b549cd9ed8dd3614652c2c2"
dataPath = "/home/kdyun/Desktop/research record/Data/"

taskname = "curriculum ablation"

# group 별로 run들을 다 log해서 mean, max, min 추출해서 band plot으로 그려주기
#input data: list(run) of dict(reward) of array(iter) -> dict(reward) of dict(run statistics) of array(iter)


trainingMode = ["100 iteration","200 iteration","300 iteration","400 iteration","1000 iteration","direct transfer",] # 이 순서로 legend 나옴
# trainingMode = ["100 iteration","200 iteration","300 iteration","400 iteration","1000 iteration"]

data_types = ['Curriculum/Total Mass', 'Curriculum/Trunk Mass', 'Rewards/groundReactionForce', 'Curriculum/Trunk Inertia',
              'Rewards/baseAngularVelError', '_runtime', 'Curriculum/(simul_frequency/1000)', 'Curriculum/MassScale',
              'Rewards/avg_reward', 'Rewards/frictionCone', 'Rewards/avg_dones', 'Validation/return', 'Rewards/standingPose',
              '_timestamp', '_step', 'Rewards/baseLinearVelError', 'Rewards/pitchVel']
# data_types = ["Validation/return",'_step']
data_types = ["Validation/return",'Curriculum/MassScale','_step']

# total logging list : should be saved in pickle file with the same name
runs = dict()
runs["direct transfer"] = ['2023-08-24-03-35-38']
runs["100 iteration"] = ['2023-08-23-03-51-53']
runs["200 iteration"] = ['2023-08-23-05-01-02']
runs["300 iteration"] = ['2023-08-23-12-53-07']
runs["400 iteration"] = ['2023-08-23-17-26-54']
runs["1000 iteration"] = ['2023-08-23-23-49-45']

# container for processed data : trainingMode(group) / data_types(categories) / statistics(in numpy array)
runStatistics = dict()
for mode in trainingMode:
    runStatistics[mode] = dict()
    for data in data_types:
        runStatistics[mode][data]= dict.fromkeys( ['mean','max','min','step'],None)
runStatistics["direct transfer"]['total iteration'] = 400
runStatistics["100 iteration"]['total iteration'] = 100
runStatistics["200 iteration"]['total iteration'] = 200
runStatistics["300 iteration"]['total iteration'] = 300
runStatistics["400 iteration"]['total iteration'] = 400
runStatistics["1000 iteration"]['total iteration'] = 200



# extract pickle file
for mode in trainingMode:
    for index in range(len(runs[mode])):
        log = loadData.load_data(path=dataPath+taskname+"/"+mode,run_name=runs[mode][index], data_types=data_types, data_length=runStatistics[mode]['total iteration'], filter_period=1)
        runs[mode][index] = log #run name string 대신 run dict로 대체

# fill in statistics
for mode in trainingMode:
    # mode ='SRB pretrained curriculum'
    print(mode)
    for data in data_types:
        print("  ",data)
        runList = []
        for index in range(len(runs[mode])):
            # print (runs[mode][index][data])
            runList.append(runs[mode][index][data])
        #calculate mean, max, min, iter됨
        runStatistics[mode][data]['mean'] = np.mean(runList,axis=0)
        runStatistics[mode][data]['max'] = np.max(runList,axis=0)
        runStatistics[mode][data]['min'] = np.min(runList,axis=0)

# print(runs)
# print(runStatistics)

#-----------------------------------------------------------------------------------------------------------------------
# plot graph
fig,ax = plt.subplots()
color_dict = dict.fromkeys(trainingMode,None)



data = "Validation/return"
for mode in trainingMode:
    # x=np.array(range(runStatistics[mode]["Validation/return"]['mean'].size))
    x= runStatistics[mode]['_step']['mean']
    # x= runStatistics[mode]['Curriculum/MassScale']['mean']
    y_mean = runStatistics[mode][data]['mean']
    # y_max = runStatistics[mode][data]['max']
    # y_min = runStatistics[mode][data]['min']
    ax.plot(x,y_mean,color= color_dict[mode],alpha =0.99 ,label= mode)
    # ax.fill_between(x, y_min, y_max,color=color_dict[mode], alpha=.5, linewidth=0)
# plt.title("curriculum speed")
# plt.xlabel("Iteration")
plt.xlabel("MassScale")
plt.ylabel("Return")
plt.legend()
# plt.xlim([0,2050])
# plt.ylim([0,65])


fig,ax = plt.subplots()
data = 'Curriculum/MassScale'
for mode in trainingMode:
    # x=np.array(range(runStatistics[mode]["Validation/return"]['mean'].size))
    x= runStatistics[mode]['_step']['mean']
    y_mean = runStatistics[mode][data]['mean']
    # y_max = runStatistics[mode][data]['max']
    # y_min = runStatistics[mode][data]['min']
    ax.plot(x,y_mean,color= color_dict[mode],label= mode)
    # ax.fill_between(x, y_min, y_max,color=color_dict[mode], alpha=.3, linewidth=0)
# plt.title("graph")
plt.xlabel("Iteration")
plt.ylabel("MassScale")
plt.legend()
# plt.xlim([0,6000])
# plt.ylim([0,65])


plt.show(block = True)
