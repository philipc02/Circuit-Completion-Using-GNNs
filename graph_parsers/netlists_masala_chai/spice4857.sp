spice
* SPICE Netlist for the given BJT amplifier circuit

VCC 8 0 DC 10
VG 5 0 AC 1

* Resistors
RG 5 2 600
R1 8 2 10k
R2 2 0 2.2k
RE 4 0 680
RC 8 3 3.6k
RLoad 3 0 270

* Capacitors (assuming ideal with no parasitics)
C1 2 5 1u
C2 4 0 1u

* NPN Transistors
Q1 3 2 4 QNPN
Q2 6 3 0 QNPN

.model QNPN NPN

.end