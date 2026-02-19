plaintext
* SPICE Netlist for the Circuit

VCC  2 0 DC
Vin  5 0 DC

RRC1 2 4 1k  ; Replace 1k with actual resistance if needed
RRC2 2 2 1k  ; Replace 1k with actual resistance if needed

Q1   4 5 0 QNPN
Q2   2 4 0 QNPN

.model QNPN NPN
.model QPNP PNP

.tran 1n 100n
.end