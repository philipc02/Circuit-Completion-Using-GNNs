spice
* SPICE netlist for the schematic

V1 4 0 DC 2.5V
RD 4 3 1k
M1 3 2 0 0 NMOS
CL 3 0 1p

* NMOS model
.model NMOS NMOS (LEVEL=1)