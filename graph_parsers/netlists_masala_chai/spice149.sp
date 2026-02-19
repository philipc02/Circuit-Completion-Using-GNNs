* Netlist for the given schematic

VDD 5 0 DC 5V
IIN 2 5 DC Iin

R1 2 5 1k

M1 4 2 6 6 NMOS W=1u L=0.18u
M2 3 2 4 4 NMOS W=1u L=0.18u

* Connections:
* VDD connected to net 5
* IIN connected between nets 2 and 5
* Resistor R connects net 2 and 5
* M1 drain to net 2, gate to net 6, source and body to net 4
* M2 drain to net 3, gate to net 2, source and body to net 4

.END