spice
* SPICE Netlist

* Voltage Source
Vin 3 2 DC

* Diode
D1 3 4 Dmodel

* Resistor
R1 4 5 1k

* Voltage Source
VB 5 2 DC

.model Dmodel D

.end