spice
* SPICE Netlist

* Voltage Source
Vg 7 0 AC 1
VCC 6 0 DC VCC

* Resistors
RG 7 2 RG
R1 6 2 R1
R2 2 4 R2
RC 6 5 RC
RE 3 8 RE
RL 5 9 RL

* Capacitors
CC1 2 3 CC1
CE1 3 8 CE1
CC2 5 9 CC2
CE2 5 8 CE2

* Transistors
Q1 3 2 8 QMOD
Q2 5 4 8 QMOD

* Model Definitions
.model QMOD NPN

.end