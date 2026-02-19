spice
* SPICE netlist
V1 9 0 DC 10
I1 9 3 DC 0.5m

* Resistors
R1 3 2 1k
R2 2 8 2k
R3 8 0 3k

* Capacitors
C1 6 7 10u
C2 8 3 5u
C3 3 0 10u

* Transistors
* PMOS: drain gate source
M1 3 6 7 7 PMOS
M2 3 10 3 3 PMOS

* NMOS: drain gate source
M3 2 4 3 3 NMOS
M4 8 1 0 0 NMOS
M5 8 5 3 3 NMOS
M6 2 2 0 0 NMOS

.model PMOS PMOS (LEVEL=1)
.model NMOS NMOS (LEVEL=1)

.end