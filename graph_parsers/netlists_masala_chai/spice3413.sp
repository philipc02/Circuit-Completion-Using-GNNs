plaintext
* SPICE Netlist for the given schematic

VCC 1 0 DC 12V
VT 5 4 DC 0V

RB1 1 2 1k
RC 2 10 4.7k
RB2 2 7 10k
RE 9 3 560
RT 5 4 500
RL 6 10 1k

CB 3 3 10u
CI 9 8 2.2u
CO 10 6 1u

Q1 9 8 3 QMOD

.model QMOD NPN(BF=100)

.END