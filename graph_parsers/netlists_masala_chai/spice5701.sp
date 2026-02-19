spice
*SPICE Netlist for the given schematic

* Diode
D1 2 3 Dmodel

* Current source
Isource 4 2 DC 1A

* Voltage measurement across diode
Vmeasure 2 3 DC 0V

* Model for Diode
.model Dmodel D

.END