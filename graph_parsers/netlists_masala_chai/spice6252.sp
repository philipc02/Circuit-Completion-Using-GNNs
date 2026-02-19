* SPICE Netlist for the provided schematic

* Voltage Sources
VCC 3 0 DC VCC
VEE 6 0 DC VEE

* BJTs
Q13 3 2 4 QNPN
Q14 5 2 7 QNPN
Q19 2 2 6 QNPN
Q18 6 8 5 QNPN
Q21 6 3 6 QPNP
Q20 6 5 6 QPNP
Q23 5 5 6 QNPN
Q17 6 3 6 QNPN

* Resistors
R10 2 3 R10_value
R5 6 6 R5_value
RL 8 6 2k

* Models
.model QNPN NPN
.model QPNP PNP

.end