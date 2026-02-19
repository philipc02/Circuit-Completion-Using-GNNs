* SPICE netlist for the given schematic

* NMOS transistors
M1 4 2 2 2 NMOS
M3 2 2 0 0 NMOS

* PMOS transistor
M2 3 4 4 4 PMOS

* Resistor
R_D 3 4 RD

* Voltage source
VDD 3 0 DC VDD
VIN 2 0 DC VIN

* Definitions
.model NMOS NMOS (level=1)
.model PMOS PMOS (level=1)

.end