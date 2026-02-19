spice
* Components
V1 6 0 DC
VS 6 5 DC 0V
RB1 2 1 1k
RB2 2 0 10k
RC 2 3 1k
RE 4 3 2k
R1 5 2 5k
R2 3 0 10k
RL 5 0 2k
RS 6 5 5k
CC1 5 2 0.00001uF
CC2 3 5 0.00001uF

* Node assignments are based on the second schematic
Q1 4 1 3 QMOD

* Model definitions
.model QMOD NPN(IS=1e-14 BF=100)

* Simulation commands
.end