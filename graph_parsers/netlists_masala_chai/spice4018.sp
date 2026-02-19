plaintext
* Circuit Netlist

V1 1 0 DC Vi
I1 2 0 DC 1.5mA
R1 2 3 5k
R2 3 0 2.8k
RL 4 0 10k
CC 2 4 
CB 3 0
Q1 4 3 1 QMODEL

.model QMODEL NPN (for example)