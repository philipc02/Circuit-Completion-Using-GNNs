spice
* SPICE Netlist for the BJT Amplifier Circuit

VCC 3 0 DC 15
VEE 2 0 DC -15
V1 5 0
V2 4 0

RC1 3 5 2k
RC2 3 6 2k
RE 2 7 2k
R1 5 4 5k
R2 5 2 10k

Q1 5 1 7 NPN
Q2 6 1 7 NPN
Q3 7 5 2 NPN

.control
run
.endc

.end