import plot_util
import torch
import numpy as np
import matplotlib.pyplot as plt

# def find_first_over_threshold(numbers, threshold):
#     for i, num in enumerate(numbers):
#         if num > threshold:
#             return i
#     return -1

# # # #Note to self: This code is meant to aggregate results from multiple runs. 
# # # #Just move all your results into one folder then specify the folder below.
# # # #Make sure to follow the follow the the folder structure (e.g .../trial_t/model_id/train_y.pt) 
# trials = 4
# env_name ="Coheritability Wavelength Ratio + Narea"
# base_path = "/lfs/ampere1/0/ruhana/bnn-bo"
# #experiment_path = "23_10_19-15_12_17_config/narea_spectral.json_narea_coh2_canceled"
# #experiment_path = "23_10_19-15_44_20_config/narea_spectral.json_narea_coh2_canceled"
# #experiment_path = "23_10_19-15_49_31_config/narea_spectral.json_narea_coh2_canceled"
# #experiment_path = "23_10_19-15_53_19_config/narea_spectral.json_narea_coh2_canceled"
# #experiment_path = "experiment_results/SLA_GP_Kernel_Test_CPU/aggregate"
# experiment_path = "experiment_results/narea_dkl/aggregate"
# #experiment_path = "SLA/aggregate" #"23_10_19-02_05_58_config/ackley.json_ackley_10_done"
# models = ["random", "gp", "dkl"] #"gp", "gp_matern", "gp_sp10"]

# def load_model(model_id):
#     Y = []
#     for trial in range(trials):
#         filex = f"{base_path}/{experiment_path}/trial_{trial+1}/{model_id}/train_x.pt"
#         filey = f"{base_path}/{experiment_path}/trial_{trial+1}/{model_id}/train_y.pt"
#         Y_i = torch.load(filey)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
#         Y.append(plot_util.getRunningBest(Y_i))
#         print(find_first_over_threshold(Y_i, 0.4103))
#     Y = torch.stack(Y)
#     return Y

# results = []
# for model_id in models:
#     results.append(load_model(model_id))

# plot_util.multiPlot(results, models, f"{env_name}", save=f"{base_path}/{experiment_path}/plot.png")
# #plt.plot(np.arange(0, 60, 1, dtype=int), [0.4103]*60, linestyle='--', color='red', label='top-1%')
# plt.savefig(f"./{experiment_path}/plot.png")

import pandas as pd

trait = "ps"
data = pd.read_csv(f"./test_functions/{trait}_coh2.csv")
data = torch.tensor(data.values, dtype=torch.float64)
# fig, ax = plt.subplots(1,1, figsize=(6,5)) 

lookup = data
x_starter, x_end = 350, 2500
y_starter, y_end = 350, 2500
x_num = data.shape[0]
y_num = data.shape[1]

# interval = lambda starter, end, num: int( (end-starter) / (num-1) )
# x_i = interval(x_starter, x_end, x_num)
# y_i = interval(y_starter, y_end, y_num)

# size_ratio = (y_end-y_starter) / (x_end-x_starter) 

# y, x = np.mgrid [ slice(y_starter, y_end + y_i, y_i ), slice( x_starter, x_end + x_i, x_i ) ] 
# cax = ax.pcolormesh(x, y, lookup, cmap = 'viridis', vmin = 0, vmax=1)

# fig.colorbar(cax, ax=ax)

# # # #plot top-1%-----------------------------------------------------------
percentage = 0.01
num_elements = int(percentage * data.numel())

# Flatten the tensor and get the indices of the top 1% elements
data_flat = data.flatten()
no_nans = data_flat[~torch.isnan(data_flat)]
_, top_indices = torch.topk(no_nans, num_elements)
threshold = no_nans[top_indices[-1]] 
print(threshold)

# y_k, x_k = np.where(lookup > threshold)
# ax.scatter(x_k + x_starter, y_k + y_starter, color="orange", s=0.001, marker='x')

# searched = torch.load(f"./experiment_results/SLA_GP_Kernel_Test_CPU/aggregate/trial_1/random/train_x.pt")
# ax.scatter(searched[:,0], searched[:,1])

# #set plot labels -----------------------------------------------------------
# #ax.set_title(f'Wavelength Pair + {trait} Co-heritability', fontsize=13)
# ax.set_title(f'Wavelength Ratio Heritability + SLA', fontsize=13)
# ax.set_xlabel("wavelength 1")
# ax.set_ylabel("wavelength 2")

# plt.savefig("test.png")              
