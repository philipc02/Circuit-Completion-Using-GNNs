spice
* SPICE Netlist for the given circuit

V1 9 0 DC 10V
V2 4 0 DC 10V
I1 0 2 DC 1A

RC 0 8 10k
RC 6 7 10k
R1 0 3 1k

Q1 3 9 8 QN
Q2 6 5 8 QN
Q3 3 3 7 QN
Q4 2 4 7 QN

.model QN NPN(IS=1E-14 BF=100)

*.end