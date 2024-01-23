import torch
import numpy as np
import matplotlib.pyplot as plt

def ismonotonic(numbers):
    return all(numbers[i] <= numbers[i + 1] for i in range(len(numbers) - 1))

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

def multiPlot(results, labels, title, save=None):
    print(f"Plotting {len(results)} seeds...")

    colors = ["Pink", "Blue", "Green", "Orange", "Purple", "Yellow"]
    for i, Y in enumerate(results):
        line = torch.arange(0, Y.shape[1], 1)
        mean, std = getStat(Y)
        #check mean is monotonically increasing
        assert ismonotonic(mean), "Your results are not monotonically increasing, this might be an issue"
        
        plt.plot(line, mean, label= labels[i], color=colors[i])
        plt.fill_between(line, mean - std,  mean + std, alpha=0.3, color=colors[i])
    plt.title(f"{title}")
    plt.xlabel("Iteration (n)")
    plt.ylabel("Max(f(X))")
    plt.legend(loc="lower right", )

    if save is not None:
        plt.savefig(save)
    return