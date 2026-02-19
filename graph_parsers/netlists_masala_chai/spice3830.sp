plaintext
* SPICE Netlist
V1 2 0 DC Vi
I1 2 1 Ii

* Resistors
RD 0 3 8k
RG 2 4 50k
RS1 3 7 100
RS2 7 0 250
RL 5 6 4k

* Capacitors
CC 2 3 CC_value
CC2 3 5 CC2_value
CS 7 0 CS_value

* Connections
VDD 0
VO 6