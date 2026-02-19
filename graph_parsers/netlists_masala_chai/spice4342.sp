spice
* Transistors
Q1 N1 N2 4 N1 NPN
Q2 N3 N5 4 N3 NPN

* Resistors
RC1 6 N1 100k
RC2 6 N3 100k
R1 N2 N4 2k
R2 N4 N5 2k
RE 4 0 85k

* Voltage Sources
VCC 6 0 DC 5
VEE 4 0 DC -5

* Nodes
V1 N2 0 
VOUT N5 0

.END