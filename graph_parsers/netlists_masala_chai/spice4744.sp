spice
* SPICE Netlist for BJT Amplifier Circuit

Q1 2 6 8 NPN_BJT
.model NPN_BJT NPN (BF=100)

RC 2 3 1k
RE 8 4 2.2k

VCC 3 4 DC 15V
VBB 6 5 DC 5V

* Defining ground
.node 4 0
.node 5 0

.end