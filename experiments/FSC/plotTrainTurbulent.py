import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import re
import sys
import os

parser = ap.ArgumentParser()
parser.add_argument("--TD", action="store_true", default=False)
parser.add_argument("--subfolder", default="tmp")
args = parser.parse_args()
sb = args.subfolder
td = args.TD

reg = re.compile(r"[-+]?\d*\.\d+") # Maybe not completely correct

folder = f"plots/{"AC" if td else "reinforceTrain"}/{sb}"
os.makedirs(folder, exist_ok=True)
killed = 0
errors = 0
total = 0
finished = 0
rollbacks = 0
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
    counted = False
    for line in open(file[:-1]):
        if line.startswith("Episode"):
            num = reg.findall(line)
            success.append(float(num[0]))
            avgSteps.append(float(num[1]))
        if "rollback" in line:
            rollbacks += 1 # Don't set count because up to 3 rollbacks are allowed during training (olny with TDAC)
        if ("Error" in line or "Determinism" in line ) and not counted:
            errors += 1
            counted = True
        elif "Terminated" in line and not counted:
            killed += 1 
            counted = True
        elif "Total" in line and not counted:
            finished += 1
            counted = True
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
    print(f"{sb} lr {lr:.2e} {M}:\tOn {total} runs: {finished} completed; {errors} reached determinism; {killed} were interrupted; {rollbacks} has been rollbacked", file=recap, end="")