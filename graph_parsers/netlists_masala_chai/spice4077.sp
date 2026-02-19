spice
* Components
V1 7 1 DC VIN
RC 5 2 RC
RL 2 1 RL
R1 5 4 R1
R2 4 3 R2
R3 7 6 R3
RE 3 6 RE
RS 7 1 RS

* BJTs
Q1 3 4 6 QNPN
Q2 2 4 3 QNPN

* Capacitors
CB 4 1 CB
CC1 4 3 C1
CC2 2 4 C2
CE 6 1 CE
CL 2 1 CL

* .model declaration
.model QNPN NPN
.model QPNP PNP