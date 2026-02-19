spice
* SPICE netlist for the given schematic

VCC 7 0 DC 5
VEE 0 2 DC 5
VIN IN 5 AC 1

CC IN 5 1u
R2 5 7 10k
R3 6 2 10k
R 7 3 10k
RL 4 0 1k

* BJTs
* NPN
Q1 4 5 6 QMODEL

* PNP
Q2 7 5 4 QMODEL
Q3 2 4 2 QMODEL
Q4 2 3 2 QMODEL

.model QMODEL NPN (IS=1e-15 BF=100)
.model QMODEL PNP (IS=1e-15 BF=100)

.END