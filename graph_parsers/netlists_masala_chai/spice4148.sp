spice
* SPICE Netlist for the given schematic

* Voltage Sources
VCC 3 0 DC VCC
Vi 5 0 AC 1

* Resistors
R1 3 4 R1
R2 9 10 R2
RE 8 0 RE
RL 2 0 RL
Ri 5 6 Ri

* Capacitors
C1 6 9 C1
C2 4 2 C2
CE 8 0 CE

* Inductor
L 4 7 L

* Transistor
Q1 4 9 8 QMOD

* Models
.model QMOD NPN

.end