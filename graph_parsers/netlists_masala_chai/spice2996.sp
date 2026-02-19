spice
* SPICE Netlist for the given schematic

* MOSFET Definitions
M1 4 3 1 1 NMOS
M2 2 3 4 2 PMOS

* Current Sources
I1 2 4 DC <value>
IZin 3 1 DC <value>

* Resistor
RS Vin 3 <value>

* Voltage Source
VDD 2 0 DC <value>

* .MODEL statements are required
.model NMOS NMOS(level=1)
.model PMOS PMOS(level=1)