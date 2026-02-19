spice
* SPICE Netlist
V1 9 0 DC 0V
V2 5 0 DC 0V
VCC 4 0 DC 10V
VEE 8 0 DC -10V

RC1 4 2 50k
RC2 4 3 50k
RB1 1 9 0.5k
RB2 3 5 0.5k
RE 2 8 RE_value

Q1 4 1 2 NPN
Q2 4 3 2 NPN

.model NPN NPN (Is=1e-15 Vaf=100 Bf=100)

.end