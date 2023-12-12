import matplotlib
import matplotlib.pyplot as plt

import pickle
import numpy as np
import loadData
import os
import copy

os.environ["WANDB_API_KEY"] = "5804d2713f53ff0c9b549cd9ed8dd3614652c2c2"
dataPath = "/home/kdyun/Desktop/research record/Data/"

taskname = "backflip"

# group 별로 run들을 다 log해서 mean, max, min 추출해서 band plot으로 그려주기
#input data: list(run) of dict(reward) of array(iter) -> dict(reward) of dict(run statistics) of array(iter)

# trainingMode = ["SRB pretraining","vanilla full","SRB pretrained full","SRB pretrained curriculum"] # 이 순서로 legend 나옴
# trainingMode = ["vanilla full"]
# trainingMode = ["SRB pretrained full","vanilla full"]
trainingMode = ["SRB pretrained curriculum","SRB pretrained full","vanilla full",]

data_types = ['Curriculum/Total Mass', 'Curriculum/Trunk Mass', 'Rewards/groundReactionForce', 'Curriculum/Trunk Inertia',
              'Rewards/baseAngularVelError', '_runtime', 'Curriculum/(simul_frequency/1000)', 'Curriculum/MassScale',
              'Rewards/avg_reward', 'Rewards/frictionCone', 'Rewards/avg_dones', 'Validation/return', 'Rewards/standingPose',
              '_timestamp', '_step', 'Rewards/baseLinearVelError', 'Rewards/pitchVel']
data_types = ["Validation/return",'_step']
# data_types = ["Validation/return",'Curriculum/MassScale','_step']

# total logging list : should be saved in pickle file with the same name
runs = dict()
runs['SRB pretraining'] = ['2023-07-31-22-33-57','2023-08-01-22-51-08', '2023-08-01-23-45-07','2023-08-02-00-38-57','2023-08-02-01-33-56']
runs['vanilla full'] = ['2023-08-07-18-36-39','2023-08-08-02-36-05','2023-08-08-12-56-52','2023-08-09-02-16-50','2023-08-09-05-33-11' ]
runs['SRB pretrained full'] = ['2023-08-01-07-55-19','2023-08-01-06-54-18','2023-08-01-05-53-47','2023-08-01-04-53-19','2023-08-01-03-53-10' ]
# runs['SRB pretrained curriculum'] = ['2023-08-23-02-55-17','2023-08-23-03-56-14','2023-08-23-04-57-26','2023-08-23-05-59-10','2023-08-23-07-00-52' ] #set2
runs['SRB pretrained curriculum'] = ['2023-08-23-15-42-57','2023-08-23-13-57-31','2023-08-23-12-49-42','2023-08-23-11-19-06','2023-08-23-02-55-17' ] #set3

# container for processed data : trainingMode(group) / data_types(categories) / statistics(in numpy array)
runStatistics = dict()
for mode in trainingMode:
    runStatistics[mode] = dict()
    for data in data_types:
        runStatistics[mode][data]= dict.fromkeys( ['mean','max','min','step'],None)
# runStatistics['SRB pretraining']['total iteration'] = 100 # 50*100 =5000
runStatistics['SRB pretrained curriculum']['total iteration'] = 200 #5*200 =1000 (500번 curriculum + 500 full)
runStatistics['SRB pretrained full']['total iteration'] = 80 # 50*80 =4000
runStatistics['vanilla full']['total iteration'] = 120 # 50*138 =6800


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
# color_dict['SRB pretraining'] = 'green'
# color_dict['SRB pretrained curriculum'] = 'tab:red'
# color_dict['SRB pretrained full'] = 'tab:blue'
# color_dict['vanilla full'] = 'tab:brown'

color_dict['SRB pretrained curriculum'] = 'tab:red'
color_dict['SRB pretrained full'] = 'tab:blue'
color_dict['vanilla full'] = 'tab:brown'


alpha_dict = dict.fromkeys(trainingMode,None)
alpha_dict['SRB pretraining'] = 0.5
alpha_dict['SRB pretrained curriculum'] = 0.5
alpha_dict['SRB pretrained full'] = 0.5
alpha_dict['vanilla full'] = 0.2


data = "Validation/return"


for mode in trainingMode:
    if mode == 'SRB pretrained curriculum':
        x= runStatistics['SRB pretrained curriculum']['_step']['mean']
        y_mean = runStatistics['SRB pretrained curriculum'][data]['mean']
        y_max = runStatistics['SRB pretrained curriculum'][data]['max']
        y_min = runStatistics['SRB pretrained curriculum'][data]['min']
        ax.plot(x[:26],y_mean[:26],color= color_dict['SRB pretrained curriculum'], alpha = 1,label ='SRB pretrained curriculum')
        ax.fill_between(x[:26], y_min[:26], y_max[:26],color=color_dict['SRB pretrained curriculum'], alpha=0.6, linewidth=0)
        ax.plot(x[25:],y_mean[25:],color= color_dict['SRB pretrained curriculum'], alpha = 0.3)
        ax.fill_between(x[25:], y_min[25:], y_max[25:],color=color_dict['SRB pretrained curriculum'], alpha=0.1, linewidth=0)
    else:
        # x=np.array(range(runStatistics[mode]["Validation/return"]['mean'].size))
        x= runStatistics[mode]['_step']['mean']
        y_mean = runStatistics[mode][data]['mean']
        y_max = runStatistics[mode][data]['max']
        y_min = runStatistics[mode][data]['min']
        ax.plot(x,y_mean,color= color_dict[mode],alpha =1 ,label= mode)
        ax.fill_between(x, y_min, y_max,color=color_dict[mode], alpha=alpha_dict[mode], linewidth=0)

# plt.title("comparison of returns")
plt.xlabel("Iteration")
plt.ylabel("Return")
# plt.legend()
plt.xlim([0,6000])
plt.ylim([0,65])


# fig,ax = plt.subplots()
# data = 'Curriculum/MassScale'
# for mode in trainingMode:
#     if mode == 'SRB pretrained curriculum':
#         # x=np.array(range(runStatistics[mode]["Validation/return"]['mean'].size))
#         x= runStatistics[mode]['_step']['mean']
#         y_mean = runStatistics[mode][data]['mean']
#         y_max = runStatistics[mode][data]['max']
#         y_min = runStatistics[mode][data]['min']
#         ax.plot(x,y_mean,color= color_dict[mode],label= mode)
#         ax.fill_between(x, y_min, y_max,color=color_dict[mode], alpha=.3, linewidth=0)
# plt.title("graph")
# plt.xlabel("iteration")
# plt.ylabel("return")
# plt.legend()
# plt.xlim([0,6000])
# # plt.ylim([0,65])


plt.show(block = True)
