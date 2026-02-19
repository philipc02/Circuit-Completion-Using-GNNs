spice
* SPICE Netlist for the given schematic

* Voltage Sources
V1 6 0 DC 5
V2 7 0 DC -5

* BJTs
Q1 3 1 4 QMODEL
Q2 5 2 7 QMODEL
Q3 6 2 3 QMODEL

* Resistors
R1 4 1 1k
R2 6 3 1k
R3 6 5 1k
R4 7 5 1k
R5 6 2 1k
R6 3 2 1k

* Models
.model QMODEL NPN

* End of Netlist