spice
* SPICE Netlist for the given schematic

V1 2 4 DC 1V
V2 2 3 DC 0.4V
V3 2 0 DC 0.9V

* NMOS transistor M1 (Drain, Gate, Source)
M1 2 2 4 4 NMOS_MODEL

* Model parameters for NMOS
.model NMOS_MODEL NMOS (LEVEL=1)

.end