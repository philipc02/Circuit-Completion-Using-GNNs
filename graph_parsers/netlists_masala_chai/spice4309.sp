plain
* SPICE netlist for the circuit

IREF 4 V+ DC  <value> ; Define the value for the reference current

M1 2 2 2 2 NMOS ; NMOS M1 with D=2, G=2, S=2, B=2
M2 3 2 2 2 NMOS ; NMOS M2 with D=3, G=2, S=2, B=2
M3 2 4 4 4 PMOS ; PMOS M3 with D=2, G=4, S=4, B=4
M4 6 4 4 4 PMOS ; PMOS M4 with D=6, G=4, S=4, B=4

V+ 4 0 DC ; Define V+ supply
V- 2 0 DC ; Define V- supply

.end