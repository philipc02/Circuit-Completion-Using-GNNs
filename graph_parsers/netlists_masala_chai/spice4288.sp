spice
* SPICE netlist for the schematic

* Current Source
IREF 2 4 DC 0

* Transistors
* NMOS: M1, M2
M1 4 6 4 4 NMOS
M2 2 2 6 4 NMOS

* PMOS: M3, M4
M3 2 2 3 3 PMOS
M4 3 2 5 5 PMOS

* Voltage Supply
VDD 3 0 DC VDD_value

* Define your models
.model NMOS NMOS
.model PMOS PMOS

.end