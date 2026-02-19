spice
* SPICE Netlist for the Schematic

* Voltage Sources
V1 8 0 DC V_T
VCC 6 0 DC V_CC
VEE 7 0 DC V_EE

* BJTs
Q1 4 9 7 QNPN
Q2 3 4 7 QNPN
Q3 2 3 5 QNPN

* Resistors
RT 8 9 R_T
RB1 9 6 R_B1
RB2 9 7 R_B2
RC1 6 4 R_C1
RC2 6 3 R_C2
RE 4 7 R_E
RE3 3 2 R_E3
RL 2 5 R_L
RB4 6 5 R_B4

* Capacitors
CE1 4 7 C_E1
CE2 3 7 C_E2

* Models
.model QNPN NPN

* End of Netlist