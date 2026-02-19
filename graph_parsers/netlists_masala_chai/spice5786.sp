spice
* SPICE Netlist for the given circuit

* Voltage Sources
V1 4 0 DC 3
V2 5 0 DC -3

* Resistors
R1 4 3 100k
R2 1 0 9.1k
R3 3 0 9.1k
R4 5 2 5.1k
R5 2 3 4.3k

* Transistors (NPN BJTs)
Q1 3 2 1 QNPN
Q2 3 2 5 QNPN

* Model Definitions
.model QNPN NPN (IS=1e-14 BF=100)

.end