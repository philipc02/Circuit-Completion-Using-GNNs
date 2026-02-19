spice
* SPICE Netlist for the given schematic

V1 v1 8 DC 0
Vplus 2 0 DC Vplus

* MOSFETs (Mname Drain Gate Source Body Model L W)
M0 3 8 7 7 NMOS L=1u W=1u
M1 4 2 5 5 PMOS L=1u W=1u
M2 4 2 6 6 PMOS L=1u W=1u
M3 5 3 5 5 NMOS L=1u W=1u

* Current Source
IREF 5 5 DC Iref

* Model Definitions
.model NMOS NMOS (LEVEL=1)
.model PMOS PMOS (LEVEL=1)

.end