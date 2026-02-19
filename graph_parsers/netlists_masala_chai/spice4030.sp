plaintext
* SPICE netlist for the given circuit
VCC 8 0 DC 10
Vs 5 0 AC 1

RIS 9 5 100k
R1 9 8 335k
R2 9 1 125k
RC 6 8 2.2k
RE2 2 7 1k
Ro 4 0 {Ro_value}

CC 5 6 {CC_value}
CE 2 0 {CE_value}

Q1 6 9 1 QNPN
Q2 4 6 7 QNPN

.model QNPN NPN