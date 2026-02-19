plaintext
* Diode Circuit SPICE Netlist
* Voltage Sources
V1 2 0 DC 3
V2 3 0 DC 2
V3 7 0 DC 1
V4 5 0 DC 5

* Resistor
R1 5 8 1k

* Diodes
D1 2 8 Dmodel
D2 3 4 Dmodel
D3 7 6 Dmodel

* Ground
V5 6 0 DC 0

* Models
.model Dmodel D

.end