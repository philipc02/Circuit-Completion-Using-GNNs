spice
* SPICE Netlist for the circuit

VCC 6 2 DC 5V
R1 6 6 10k
R2 6 7 10k
Q1 6 3 2 QNPN

.model QNPN NPN (BF=100)

* Connections
* Node 2 is Ground
* Node 3 is input v1
* Node 6 is +5V
* Node 7 is output vo

.END