spice
* SPICE netlist for given schematic

V1 6 0 DC VGSGQ
IREF 2 0 DC 80uA

* NMOS transistors
M1 4 6 0 0 NMOS
M2 3 4 6 6 NMOS
M3 2 3 0 0 NMOS

* PMOS transistors
M4 5 2 5 5 PMOS
M5 2 5 V+ V+ PMOS
M6 3 2 V+ V+ PMOS

* Voltage source
V_SUPPLY V+ 0 DC 9V

.model NMOS NMOS
.model PMOS PMOS

.end