spice
* Netlist for the given schematic

* NMOS Transistors
M1 3 1 4 4 NMOS
M2 6 8 4 4 NMOS

* PMOS Transistors
M3 5 3 5 5 PMOS
M4 5 2 5 5 PMOS

* Current Source
I1 4 7 DC IQ

* Voltage Definitions
V+ 5 0 DC V+
V- 4 0 DC V-

* Other Nodes
V1 1 0 DC V1
V2 2 0 DC V2
VO 8 0

.end