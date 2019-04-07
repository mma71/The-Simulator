# The-Simulator
A discrete-time event simulator for a number of CPU schedulers on a single CPU system. The goal of this project is to compare and assess the impact of different schedulers on different performance metrics, and across multiple workloads.

Implemented are the following scheduling algorithms:
1. First-Come First-Served (FCFS)
2. Shortest Remaining Time First (SRTF)
3. Highest Response Ratio Next (HRRN)
4. Round Robin, with different quantum values (RR)

Interested to compute the following metrics, for each experiment: 
- The average turnaround time
- The total throughput (number of processes done per unit time) 
- The CPU utilization

Instructions:
To run from terminal:
python code.py algo lambda svcs quantum


***Please Note: code is written in Python 3***
