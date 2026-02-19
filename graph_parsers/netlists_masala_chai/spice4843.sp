plaintext
*SPICE Netlist

VCC 8 0 DC 10
Vin 1 0 AC 1

*Transistors
Q1 2 5 7 QNPN
Q2 6 8 7 QNPN

*Resistors
R1 2 3 10k
R2 3 1 2.2k
RC1 5 8 3.6k
R3 5 7 180
RE1 7 2 820
RC2 6 8 3.6k
R4 9 10 2.2k
RE2 10 7 820
Rf 10 7 5

*Capacitors
C1 3 5 1u
C2 5 6 1u
C3 6 8 1u
C4 9 11 1u

.model QNPN NPN

.end