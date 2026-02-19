spice
* Netlist for the BJT Amplifier Circuit

VCC 7 0 DC 0
Vo 4 0 AC 1

RG 4 2 1k
R1 7 2 10k
RC 7 3 1k
RE 3 0 560
RL 8 0 1k
R2 2 0 5k

Cin 2 22 10u
Cout 3 8 10u
CE 3 0 100u

Q1 7 22 3 QMOD

.model QMOD NPN(IS=1e-15 BF=100)

.end