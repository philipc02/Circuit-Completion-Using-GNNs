plaintext
* SPICE Netlist

Q1 6 3 2 NPN

RB1 3 2 1k
RB2 3 2 1k
RC 6 2 1k

IC 6 2 DC 1A
VCC 6 2 DC 10V

.model NPN NPN(IS=1e-14 BF=100)

.end