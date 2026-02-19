spice
* SPICE Netlist for the given schematic

* Voltage Source
V1 2 3 DC Vin

* Resistor
RT 2 4 RT

* Current Source
I1 5 3 DC Gm*Vin

* Resistor roD
RoD 5 3 roD

* Resistor roL + RL
RoL_RL 6 3 roL_plus_RL

* Connections:
* Node 4 is a common node for RT 
* Node 3 is the common ground connection

.end