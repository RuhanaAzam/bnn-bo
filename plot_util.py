import matplotlib.pyplot as plt
import torch 

def getRunningBest(X):
    running_best_values = torch.empty(len(X))
    running_best = X[0]

    # Loop through the elements of X
    for i in range(len(X)):
        if X[i] > running_best:
            running_best = X[i]  # Update the running best value
        running_best_values[i] = running_best
    return running_best_values #torch list

def getStat(X):
    mean, std = torch.mean(X, axis=0), torch.std(X, axis=0)
    return mean.reshape(-1), std.reshape(-1)

def multiPlot(results, labels, title, save="./test.png"):
    print(len(results))
    colors = ["Pink", "Blue", "Green", "Orange", "Purple", "Yellow"]
    for i, Y in enumerate(results):
        line = torch.arange(0, Y.shape[1], 1)
        mean, std = getStat(Y)
        plt.plot(line, mean, label= labels[i], color=colors[i])
        plt.fill_between(line, mean - std,  mean + std, alpha=0.3, color=colors[i])
    plt.title(f"{title}")
    plt.xlabel("Iteration (n)")
    plt.ylabel("Max(f(X))")
    plt.legend(loc="lower right", )
    plt.savefig(save)
    return 

# Y = []
# for trial in range(1,6):
#     filex = f"/lfs/ampere1/0/ruhana/bnn-bo/experiment_results/23_09_27-10_42_54_config/small_test.json_ackley_10_done/trial_{trial}/gp/train_x.pt"
#     filey = f"/lfs/ampere1/0/ruhana/bnn-bo/experiment_results/23_09_27-10_42_54_config/small_test.json_ackley_10_done/trial_{trial}/gp/train_y.pt"
#     Y_i = torch.load(filey)
#     Y.append(getRunningBest(Y_i))
# Y = torch.stack(Y)
# multiPlot([Y], ["gp"], "Title Here")
# exit()
