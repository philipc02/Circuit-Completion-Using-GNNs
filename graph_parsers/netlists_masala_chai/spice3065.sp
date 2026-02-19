spice
* SPICE netlist for the schematic

VDD 2 0 DC VDD
Vin 9 8 DC Vin

M1 3 9 8 8 NMOS
M2 3 7 7 7 NMOS
M3 4 6 2 2 PMOS
M4 4 5 2 2 PMOS

RF_left 4 4 1k
RF_right 4 4 1k

Iss 3 0 DC Iss

* Additional details
.model NMOS NMOS level=1
.model PMOS PMOS level=1

.end