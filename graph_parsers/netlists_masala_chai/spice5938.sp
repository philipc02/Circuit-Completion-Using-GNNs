* SPICE netlist for the given schematic

IREF 0 8 DC 1A

* NMOS Transistors
M1 4 5 7 7 NMOS
M2 3 2 7 7 NMOS
M3 4 3 7 7 NMOS

* PMOS Transistor
M4 8 4 2 2 PMOS

* Voltage source for V_G
V1 4 0 DC 1V

* Models
.model NMOS NMOS
.model PMOS PMOS

.end