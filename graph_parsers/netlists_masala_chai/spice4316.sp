spice
* Voltage sources
V1 4 0 DC
V2 6 0 DC
Vplus 5 0 DC
Vminus 10 0 DC

* Current source
Iq 10 0 DC

* NMOS transistors (assuming all are NMOS)
M1 3 4 10 10 NMOS
M2 6 4 10 10 NMOS
M3 3 9 5 5 NMOS
M4 7 6 5 5 NMOS

* PMOS transistors
M5 9 2 5 5 PMOS
M6 8 5 5 5 PMOS

.model NMOS NMOS (LEVEL=1)
.model PMOS PMOS (LEVEL=1)

.end