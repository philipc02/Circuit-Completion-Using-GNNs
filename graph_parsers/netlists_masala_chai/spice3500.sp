plaintext
* SPICE Netlist
* Resistor and cap values are placeholders and need to be adjusted per design specifics
VCC 5 0 DC VCC
VIN 8 0 AC 1

RT 8 6 1k
R11 5 4 10k
RC1 4 5 10k
RC2 4 5 10k
R21 3 2 10k
RE1 2 222 1k
RE2 2 222 1k
RL 2 0 1k

C1 7 6 1u
CE1 2 222 1u
CE2 2 222 1u
CO 2 222 1u

Q1 2 3 222 NPN
Q2 2 3 222 NPN

* End of Netlist