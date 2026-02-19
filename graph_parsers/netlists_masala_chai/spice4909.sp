spice
* SPICE Netlist for the given Schematic

VCC 1 0 DC 20
VIN 5 0 AC 1

R1 6 12 3.9k
R2 2 0 3.9k
RL 3 4 10

Q1 1 6 10 NPN
Q2 3 2 9 NPN

D1 8 5 D
D2 2 8 D

.model NPN NPN
.model D D

.end