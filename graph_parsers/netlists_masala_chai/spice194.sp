spice
* SPICE Netlist for the given schematic

* Voltage Sources
Vi 7 3 DC 1V
VCC 4 0 DC 15V
NEG_VCC 0 6 DC -15V

* Transistors
Q1 3 7 3 NPN
Q2 5 2 4 NPN
Q3 3 5 3 NPN
Q4 6 3 5 NPN

* Diodes
D1 2 4 D

* Resistor
R1 2 4 1k

* Load Resistor
RL 5 0 1k

* Model definitions
.model NPN NPN(Is=1e-14 VAF=100 BF=100)
.model D D(Is=1e-14)