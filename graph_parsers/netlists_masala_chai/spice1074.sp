plaintext
* SPICE Netlist
VCC 6 0 DC 2.5

R1 6 2 1k
R2 2 0 1k
RC 6 3 1k
RE 2 4 1k
CB 1 2 10u

Q1 3 2 4 QMODEL

.model QMODEL NPN(IS=1e-14 BF=100)