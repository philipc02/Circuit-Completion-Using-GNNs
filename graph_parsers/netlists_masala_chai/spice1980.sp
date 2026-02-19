spice
* SPICE netlist

* NPN Transistors
Q1 4 3 5 NPN 
Q2 5 7 2 NPN

* Current Source
I1 3 4 DC 1mA

* Diodes
D1 3 6 Dmodel
D2 6 7 Dmodel

* Resistor
RL 5 0 1k

* Voltage Sources
VCC 3 0 DC 15V
VEE 2 0 DC -15V

.model Dmodel D

* End of netlist