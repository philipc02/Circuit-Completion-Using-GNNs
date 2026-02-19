spice
* SPICE Netlist for Current Mirror Amplifier
.model QNPN NPN

* Transistors
QREF net2 net1 0 QNPN
Q1 net2 net3 0 QNPN

* Current Sources
Iin net1 0 DC 1mA
Iout net3 0 DC 1mA

* Connections
* net1: Top node connected to Iin
* net2: Common node for the collectors of QREF and Q1
* net3: Emitter of Q1 connected to Iout

.end