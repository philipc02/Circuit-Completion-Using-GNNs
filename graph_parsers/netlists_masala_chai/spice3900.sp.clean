plaintext
* SPICE netlist for given schematic

VCC 2 0 DC 5V

* Transistors
Q1 6 4 5 QNPN
Q2 3 4 3 QNPN

* Current Sources
I1 5 0 DC 10I
I2 3 0 DC I

* Models for the NPN transistors
.model QNPN NPN (IS=1e-14 BF=100)

* Nodes
* 1 - Base of Q1 and Q2 (Connected to VCC)
* 2 - VCC
* 3 - Collector of Q2, connected to I
* 4 - Common base of Q1 and Q2
* 5 - Collector of Q1
* 6 - Emitter of Q1 connected to 10I

.end