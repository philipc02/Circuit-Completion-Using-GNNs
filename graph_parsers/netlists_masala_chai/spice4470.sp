spice
* SPICE netlist for the given schematic

* Current Sources
I1 3 6 DC IBias
I2 4 5 DC IBias

* Diodes
D1 3 9 Dmodel
D2 9 2 Dmodel

* NMOS Transistor
M1 8 9 6 6 NMOSMODEL

* PMOS Transistor
M2 8 9 5 5 PMOSMODEL

* Resistor
RL 8 vO RL

* Voltage Sources
V1 vI 2 DC V1
V2 V+ 6 DC V+
V3 V- 5 DC V-

* Model Declarations
.model Dmodel D
.model NMOSMODEL NMOS
.model PMOSMODEL PMOS