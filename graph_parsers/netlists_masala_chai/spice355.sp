plaintext
* SPICE Netlist
V1 1 0 DC Vs
I1 2 0 DC 0 ; Dependent current source is treated as a placeholder

Q1 3 2 0 QNPNmodel
Q2 4 3 0 QNPNmodel
Q3 5 4 0 QNPNmodel

R_E1 3 0 RE1
R_E12 4 0 RE12
R_F 2 3 RF
R_F1 4 5 RF1
R_L1 5 6 RL1
R_L2 5 0 RL2

.model QNPNmodel NPN