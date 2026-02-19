spice
* BJT Amplifier Circuit

* Voltage Sources
Vplus 10 0 DC 9
Vi 8 0 AC 1

* Resistors
R1 2 3  R1_value
R2 2 7  R2_value
R3 7 6  R3_value
RC 3 4  RC_value
RE 7 6  RE_value
RL 4 0  RL_value

* Capacitors
CB 2 0  CB_value
CC1 8 2 CC1_value
CC2 4 5 CC2_value

* Transistors
Q1 7 2 6 QPNP
Q2 3 2 4 QNPN

* Models
.model QPNP PNP
.model QNPN NPN

.end