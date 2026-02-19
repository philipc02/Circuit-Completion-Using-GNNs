plaintext
* BJT Circuit Netlist

VCC 1 0 DC 15
VEE 8 0 DC 15

R1 1 2 3.3k
R2 1 4 3.3k
R3 6 8 2.2k

Q1 6 5 7 QMODEL
Q2 4 3 7 QMODEL

*QMODEL definition
.model QMODEL NPN

* Connections
VOUT 7 0 DC 0