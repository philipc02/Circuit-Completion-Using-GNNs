spice
* SPICE Netlist for the given schematic

* Voltage Sources
V1 1 0 DC
V2 3 0 DC
Vplus 2 0 DC 10
Vminus 5 0 DC 5

* Resistors
R1 1 2 0.5k
R2 3 4 0.5k
R3 2 4 9.5k

* Diodes
D1 2 4 Dmodel
D2 4 5 Dmodel
D3 4 5 Dmodel

* Diode Model
.model Dmodel D