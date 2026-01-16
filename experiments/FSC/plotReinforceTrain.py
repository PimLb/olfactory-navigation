import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import re
import sys
import os

parser = ap.ArgumentParser()
parser.add_argument("--subfolder", default="tmp")
args = parser.parse_args()
sb = args.subfolder

reg = re.compile(r"[-+]?\d*\.\d+") # Maybe not completely correct

folder = f"plots/reinforceTrain/{sb}"
os.makedirs(folder, exist_ok=True)
killed = 0
errors = 0
total = 0
finished = 0
for file in sys.stdin:
    total += 1
    tmp = file.split("/")
    l = tmp[-2].split("_")
    lr = float(reg.search(tmp[6]).group(0)) # Won't catch scientific notation. Not a problem for now, tho
    grad = tmp[4]
    M = tmp[5]
    n, episodes = l[:-1], l[-1]
    name = f"{n[0]}"
    for i in range(1, len(n)):
        name += f"_{n[i]}"
    name += f"_{grad}_{lr:.2e}_{M}"
    success = []
    avgSteps = []
    for line in open(file[:-1]):
        if line.startswith("Episode"):
            num = reg.findall(line)
            success.append(float(num[0]))
            avgSteps.append(float(num[1]))
        if "Error" in line:
            errors += 1
        elif "Terminated" in line:
            killed += 1 
        elif "Total" in line:
            finished += 1
    plt.figure(figsize=(20, 10))
    plt.suptitle(name)
    
    plt.subplot(1,2,1).set_title("Success Rate")
    plt.plot(range(len(success)), success, label="success percentage")
    plt.ylim(-5, 110)
    
    plt.subplot(1,2,2).set_title("Average Steps per episode")
    plt.plot(range(len(avgSteps)), avgSteps, label="Average Steps per episode")
    plt.ylim(-50, 10000)
    plt.savefig(f"{folder}/{name}.png")
    plt.close()
with open(folder+"/stats.out","w") as recap:
    print(f"{sb} lr {lr:.2e} {M}:\tOn {total} runs: {finished} completed; {errors} reached determinism; {killed} were interrupted", file=recap)