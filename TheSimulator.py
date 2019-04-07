import numpy as np
import heapq
import sys
import matplotlib.pyplot as plt

# Here we update readyQ up to the time "currentTime"
def updateReadyQ(eventQ, readyQ):
    global currentTime
    global rrid
    while eventQ:
        t, s = heapq.heappop(eventQ)
        if t <= currentTime:
            if algorithm == 0:  # FCFS
                heapq.heappush(readyQ, (t, s))
            elif algorithm == 1:  # SRTF
                heapq.heappush(readyQ, (s, t))
            elif algorithm == 2:  # HRRN
                waitingTime = currentTime - t
                estRunTime = s
                nresponse = -(1 + waitingTime / estRunTime)
                heapq.heappush(readyQ, (nresponse, t, s))
            else:  # RR
                heapq.heappush(readyQ, (rrid, t, s))
                rrid += 1
        else:
            heapq.heappush(eventQ, (t, s))  # Put it back for now
            break


if len(sys.argv) != 5:
    print('Usage: {} algorithm lambda svcs quantum')
    exit(0)

milliseconds = 0
algorithm = int(sys.argv[1]) - 1
L = float(sys.argv[2])
avgS = float(sys.argv[3])
quantum = float(sys.argv[4])
rrid = 0
algorithmNames = ['FCFS', 'SRTF', 'HRRN', 'RR']

if quantum < 0:
    print("invalid quantum: defaulting quantum to 0.01")
    quantum = 0.01
if algorithm < 0:
    algorithmList = [(i, 0.01) for i in range(4)] + [(3, 0.2)]
else:
    algorithmList = [(algorithm, quantum)]

if L < 0:
    lambdaList = list(range(1, 31))
else:
    lambdaList = [L]

# the lists to store the metrics
m1 = []
m2 = []
m3 = []
#m4 = []

for algorithm, quantum in algorithmList:
    # here we work with List of lists to store the metrics.
    m1.append([])
    m2.append([])
    m3.append([])
    #m4.append([])

    for L in lambdaList:
        NUMTOPROCESS = 10000

        # Calculate the arrival times based on the lambda "L"
        interarrivals = np.random.exponential(1.0/L, int(NUMTOPROCESS*1.1))
        arrivals = []
        t=0
        for dt in interarrivals:
            t += dt
            arrivals.append(t)

        # The service time is  chosen with the exponential distribution
        # average service time of 0.06 sec.
        servicetimes = np.random.exponential(avgS, int(NUMTOPROCESS * 1.1))

        # in order to keep track and process events in the right order we keep the events
        # in a priority queue (called â€œEvent Queueâ€) that describes the future
        # events and is sorted
        eventQ = [(arrivals[i], servicetimes[i]) for i in range(int(NUMTOPROCESS * 1.1))]
        heapq.heapify(eventQ)
        readyQ = []

        # compute the following metrics for each experiment:

        # The average turnaround time
        turnaround = []

        # The total throughput (number of processes done per unit time)
        # The CPU utilization
        idleTime = 0

        processed = 0
        # The simulator keeps a clock the represents the current
        # time which takes the time of the first event in the Event Queue.
        currentTime = 0

        print('Running {} for {} processes with lambda = {} svcs={}'.format(algorithmNames[algorithm], NUMTOPROCESS, L, avgS))

        t, s = heapq.heappop(eventQ)
        heapq.heappush(eventQ, (t, s))
        currentTime = t

        while processed < NUMTOPROCESS:
            if not readyQ:
                t, s = heapq.heappop(eventQ)
                heapq.heappush(eventQ, (t, s))
                idleTime += t - currentTime
                currentTime = t

            updateReadyQ(eventQ, readyQ)

            # Get the next process based on algorithm
            # FCFS
            if algorithm == 0:
                t, s = heapq.heappop(readyQ)
                currentTime += s
                processed += 1
                turnaround.append(currentTime - t)
            elif algorithm == 1:
                s, t = heapq.heappop(readyQ)
                currentTime += s
                processed += 1
                turnaround.append(currentTime - t)
            elif algorithm == 2:
                r, t, s = heapq.heappop(readyQ)
                currentTime += s
                processed += 1
                turnaround.append(currentTime - t)
            else:  # RR
                r, t, s = heapq.heappop(readyQ)
                if s < quantum:
                    currentTime += s
                    processed += 1
                    turnaround.append(currentTime - t)
                else:
                    currentTime += quantum
                    s -= quantum
                    updateReadyQ(eventQ, readyQ)
                    heapq.heappush(readyQ, (rrid, t, s))
                    rrid += 1

            updateReadyQ(eventQ, readyQ)

        print('Stats of {} for {} processes with lambda = {} svcs={}'.format(algorithmNames[algorithm], NUMTOPROCESS, L, avgS))
        print('Avg Turnaround: {:.4f}'.format(sum(turnaround) / float(len(turnaround))))
        print('Total throughput (number of processes done per unit time): {:.4f}'.format(NUMTOPROCESS / currentTime))
        print('CPU utilization: {:.2f}%'.format(100.0 * (currentTime - idleTime) / currentTime))
        m1[-1].append(sum(turnaround) / float(len(turnaround)))
        m2[-1].append(NUMTOPROCESS / currentTime)
        m3[-1].append(100.0 * (currentTime - idleTime) / currentTime)
        #m4[-1].append( 5)

if True:
    lines = []
    count = 0
    for sim in algorithmList:
        if sim[0] < 3 or count == 0:
            s = algorithmNames[sim[0]]
        elif sim[0] == 3 and count == 3:
            s = algorithmNames[3] + ' quantum 0.01'
        elif sim[0] == 3 and count == 4:
            s = algorithmNames[3] + ' quantum 0.2'

        line, = plt.plot(lambdaList, m1[count], label=str(s))
        lines.append(line)
        count+=1
    plt.ylabel('Time')
    plt.xlabel('Lambda')
    plt.title('Avg Turnaround')
    plt.legend(handles=lines)
    plt.show()

    lines = []
    count = 0
    for sim in algorithmList:
        if sim[0] < 3 or count == 0:
            s = algorithmNames[sim[0]]
        elif sim[0] == 3 and count == 3:
            s = algorithmNames[3] + ' quantum 0.01'
        elif sim[0] == 3 and count == 4:
            s = algorithmNames[3] + ' quantum 0.2'

        line, = plt.plot(lambdaList, m2[count], label=str(s))
        lines.append(line)
        count+=1
    plt.ylabel('Process/Time')
    plt.xlabel('Lambda')
    plt.title('Total Throughput')
    plt.legend(handles=lines)
    plt.show()

    lines = []
    count = 0
    for sim in algorithmList:
        if sim[0] < 3 or count == 0:
            s = algorithmNames[sim[0]]
        elif sim[0] == 3 and count == 3:
            s = algorithmNames[3] + ' quantum 0.01'
        elif sim[0] == 3 and count == 4:
            s = algorithmNames[3] + ' quantum 0.2'

        line, = plt.plot(lambdaList, m3[count], label=str(s))
        lines.append(line)
        count+=1
    plt.ylabel('CPU Utilization')
    plt.xlabel('Lambda')
    plt.title('CPU Utilization')
    plt.legend(handles=lines)
    plt.show()