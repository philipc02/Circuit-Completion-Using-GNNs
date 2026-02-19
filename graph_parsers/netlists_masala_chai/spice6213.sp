plaintext
* SPICE Netlist

* Voltage Sources
VCC 6 0 DC VCC
VEE 3 0 DC -VCC

* Resistor
RL 4 7 RL

* NPN Transistor
QN 4 5 6 NPN

* PNP Transistor
QP 4 5 3 PNP

* Op-Amp
* Node 5 is the input and node 4 is output based on schematic
EAMP 4 0 5 0 A0

* End of netlist