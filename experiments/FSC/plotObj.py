import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse as ap

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("name", help="name of the plot. It will be put in the title too")
    args = parser.parse_args()
    name = args.name
    plt.figure(figsize=(15, 10))
    ticks = [0, -0.1, -0.3, -0.4,-0.485, -0.6, -0.7, -0.8, -0.9, -1, -0.138, -0.197]
    totIter = 0
    for line in sys.stdin:
        obj = np.load(line[:-1])
        if obj.shape[0] > totIter:
            totIter = obj.shape[0]
        plt.plot(range(obj.shape[0]), obj,marker=None if obj.shape[0] > 5 else "x",markersize = 8)
    plt.hlines(-0.138, 0,totIter, "r", label = f"Optimal M3")
    plt.hlines(-0.197, 0,totIter, "y", label = f"Optimal M2")
    plt.hlines(-0.485, 0,totIter, "g", label = f"Optimal M1")
    plt.yticks(ticks)
    plt.ylim(-1, -0.1)
    plt.legend()
    plt.title(name)
    plt.savefig(f"plots/{name}.png")
    print("done")