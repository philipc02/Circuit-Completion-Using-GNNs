spice
* SPICE Netlist for the given circuit

VCC 4 0 DC 2.5
VEE 5 0 DC -2.5
Vin 1 0

Q1 3 4 4 PNP_MODEL
Q2 3 5 5 NPN_MODEL
R1 3 2 8

* Model Definitions
.model PNP_MODEL PNP
.model NPN_MODEL NPN

.end