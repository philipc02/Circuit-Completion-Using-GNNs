spice
* SPICE Netlist
VCC 1 4 DC 10
Vin 3 0 AC 1

* Transistors
Q1 6 3 2 QNPN
Q2 10 7 2 QNPN

* Resistors
R1 3 6 10k
R2 3 2 2.2k
R3 7 12 10k
RC1 1 6 3.6k
RC2 1 10 3.6k
RE1 2 0 1k
RE2 2 4 1k
RF 4 5 10k

* Capacitors (values need to be specified)
C1 3 5 <value>
C2 6 9 <value>
C3 10 12 <value>

.model QNPN NPN