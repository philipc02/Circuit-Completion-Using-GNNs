plaintext
* SPICE Netlist
VCC 8 1 DC 10
V_s 7 4 DC 0

* Resistors
R1 8 9 80k
R2 9 7 20k
R3 8 5 85k
R4 5 6 15k
RC1 4 8 4k
RC2 5 32 4k
RE1 2 1 1k
RE2 3 1 0.5k
RL 32 1 4k

* Capacitors
C1 9 4 4u
C2 6 2 4u
C3 32 5 4u
CE1 2 1 1u
CE2 3 1 0.5u

* Transistors
Q1 4 9 2 QNPN
Q2 5 6 3 QNPN

* Model for Q1 and Q2
.model QNPN NPN(IS=1E-14 BF=100)

.end