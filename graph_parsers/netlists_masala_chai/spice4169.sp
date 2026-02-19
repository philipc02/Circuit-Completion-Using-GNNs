spice
* SPICE Netlist for the given schematic

.IBias 4 2 DC

Q1 4 2 3 NPN
Q2 3 3 5 NPN
Q3 3 3 5 NPN

D1 4 3 Dmodel
D2 3 3 Dmodel

RL 6 5 5k

VPLUS 7 4 DC
VNEG 5 0 DC

* .model declarations
.model NPN NPN
.model Dmodel D

.END