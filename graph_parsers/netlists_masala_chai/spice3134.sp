spice
* SPICE Netlist for the given circuit

VDD 2 0 DC 15
Ib 2 3 DC 1mA
Iss 3 0 DC 1mA

M1 3 7 8 8 NMOS
M2 5 6 4 4 NMOS
M3 2 3 7 7 PMOS
M4 5 3 2 2 PMOS
M15 3 3 8 8 PMOS
M16 3 3 8 8 PMOS

* Model Definitions (Assuming basic model parameters)
.model NMOS NMOS (Level=3)
.model PMOS PMOS (Level=3)

.end