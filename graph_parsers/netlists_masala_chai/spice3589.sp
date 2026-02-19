plaintext
* SPICE Netlist

* Voltage Sources
Vi 5 2 DC 5V
Vps 6 3 DC 5V

* Diode
D1 2 4 Dmodel

* Resistor
R1 4 3 1k

.model Dmodel D
.tran 1n 10u
.end