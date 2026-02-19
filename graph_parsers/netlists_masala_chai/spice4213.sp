spice
* SPICE netlist for the given schematic

V1 6 0 DC VI
V2 3 0 DC VOB
V3 5 0 DC VOC

R1 6 6 12k
R2 6 2 12k
R3 7 6 12k
R4 2 3 40k
R5 4 3 30k
R6 7 4 12k

* Op-Amp 1
XU1 2 6 3 opamp

* Op-Amp 2
XU2 4 7 5 opamp

* Model for opamps
.model opamp opamp

.end