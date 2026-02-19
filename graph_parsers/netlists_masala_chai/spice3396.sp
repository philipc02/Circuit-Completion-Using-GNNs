spice
* SPICE Netlist for Simple BJT Circuit
VCC 3 0 DC VCC

RB 5 2 R_B
RC 4 2 R_C

Q1 4 5 6 QNPN

.model QNPN NPN (BF=100)

.END