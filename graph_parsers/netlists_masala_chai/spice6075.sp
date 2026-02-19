plaintext
* SPICE netlist for the schematic

Q1 4 1 3 NPN
Q2 5 2 3 NPN
Q3 3 1 6 NPN
Q4 3 6 3 NPN
Q5 4 3 2 NPN
Q6 2 5 5 NPN

I1 4 3 DC 0.4m
I2 5 2 DC 0.5m
I3 5 3 DC 1m

CC 3 2 1p

RL 5 0 1k

VCC 4 0 DC 5
VEE 6 0 DC -5

.END