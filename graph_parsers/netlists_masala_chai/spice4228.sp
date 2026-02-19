plaintext
* SPICE netlist for the given schematic

* Current Source
IREF 5 V+ DC

* Transistors
Q1 5 3 6 NPN
Q2 2 3 4 NPN
Q3 5 6 5 PNP
Q4 7 3 5 PNP

* Voltage supply nodes
V+ V+ 0 DC
V- 4 0 DC

* Other nodes
VBE4 7 3 DC
VBE2 3 4 DC

.end