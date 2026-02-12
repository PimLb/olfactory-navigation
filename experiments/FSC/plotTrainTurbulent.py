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
    rollbackPoint = []
    beenKilled = False
    hasFinished = False
    total += 1
    tmp = file.split("/")
    l = tmp[-2].split("_")
    if td:
        lr = reg.findall(tmp[6])
    else:
        lr = float(reg.search(tmp[6]).group(0)) # Won't catch scientific notation. Not a problem for now, tho
    lr = f"actor{float(lr[0]):.2e}_critic{float(lr[1]):.2e}" if td else f"{lr:.2e}"
    grad = tmp[4] if not td else tmp[2]
    M = tmp[5] if not td else tmp[3]
    n, episodes = l[:-1], l[-1]
    name = f"{n[0]}"
    for i in range(1, len(n)):
        name += f"_{n[i]}"
    name += f"_{grad}_{lr}_{M}"
    success = []
    timeToReach = []
    empiricalJ = []
    counted = False
    iterations = 0
    for line in open(file[:-1]):
        if line.startswith("Episode"):
            iterations += 1
            num = reg.findall(line)
            success.append(float(num[0]))
            timeToReach.append(float(num[1]))
            empiricalJ.append(float(num[2]))
        if "rollback" in line:
            rollbacks += 1 # Don't set count because up to 3 rollbacks are allowed during training (olny with TDAC)
            rollbackPoint.append(iterations)
        if ("Error" in line or "Determinism" in line ) and not counted:
            errors += 1
            counted = True
        elif "Terminated" in line and not counted:
            killed += 1 
            counted = True
            beenKilled = True
        elif "Total" in line and not counted:
            finished += 1
            counted = True
            hasFinished = True
    plt.figure(figsize=(20, 10))
    plt.suptitle(name)
    
    plt.subplot(1,3,1).set_title("Success Rate")
    plt.plot(range(len(success)), success, label="success percentage")
    plt.ylim(-5, 110)
    plt.scatter(rollbackPoint, np.array(success)[rollbackPoint], marker="x", color="r")
    if beenKilled:
        plt.scatter(len(success)-1, success[-1], marker="*", color="k")
    if hasFinished:
        plt.scatter(len(success)-1, success[-1], marker="o", color="g")

    plt.subplot(1,3,2).set_title("Time To Reach")
    plt.plot(range(len(timeToReach)), timeToReach, label="Time To Reach")
    plt.scatter(rollbackPoint, np.array(timeToReach)[rollbackPoint], marker="x", color="r")
    if beenKilled:
        plt.scatter(len(timeToReach)-1, timeToReach[-1], marker="*", color="k")
    if hasFinished:
        plt.scatter(len(timeToReach)-1, timeToReach[-1], marker="o", color="g")
    plt.ylim(0, 1)

    plt.subplot(1,3,3).set_title("Empirical J")
    plt.plot(range(len(empiricalJ)), empiricalJ, label="Empirical J")
    plt.scatter(rollbackPoint, np.array(empiricalJ)[rollbackPoint], marker="x", color="r")
    if beenKilled:
        plt.scatter(len(empiricalJ)-1, empiricalJ[-1], marker="*", color="k")
    if hasFinished:
        plt.scatter(len(empiricalJ)-1, empiricalJ[-1], marker="o", color="g")
    plt.ylim(-1, -0.1)

    plt.savefig(f"{folder}/{name}.svg")
    plt.close()
with open(folder+"/stats.out","w") as recap:
    print(f"{sb} lr {lr} {M}:\tOn {total} runs: {finished} completed; {errors} reached determinism; {killed} were interrupted; {rollbacks} has been rollbacked", file=recap, end="")