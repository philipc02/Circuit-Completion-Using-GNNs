* SPICE Netlist for the given circuit

V1 4 0 DC 5
V2 8 0 DC -5
R1 5 2 1k
RL 3 0 1k
IQ 2 8 DC 0
IL 3 0 DC 0
IE1 4 3 DC 0

* Transistors
* PMOS: Drain (D), Gate (G), Source (S)
M1 3 4 5 5 PMOS

* NMOS: Drain (D), Gate (G), Source (S)
M2 3 2 8 8 NMOS
M3 2 7 8 8 NMOS

.end