spice
* Circuit Netlist
* Transistor
Q1 2 1 3 QNPN

* Resistors
RC 6 2 RC_value
RE 3 5 RE_value

* Capacitors
C1 1 4 C1_value
C2 2 7 C2_value

* Voltage Sources
VCC 6 0 DC VCC_value
VEE 5 0 DC VEE_value

* Other Nodes
Vin 1 4 AC Vin_value
Vout 2 7

.model QNPN NPN