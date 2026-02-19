spice
* SPICE Netlist
VCC VCC 0 DC 15

* OPAMP
U1 0 Vi 2 3 OPAMP

* Transistor Q1
Q1 2 2 3 NPN

* Resistors
RL 2 3 1k
RE 3 4 1k

* Voltage Sources
Vi Vi 0 DC 5
Vfb 4 0 DC 0

.model NPN NPN
.model OPAMP opamp

.end