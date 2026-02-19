spice
* SPICE Netlist for Amplifier Circuit

Vsig 5 7 DC 0V AC 1V
Rsig 3 5 1k
Rc 3 3 1k
RL 6 1 1k

* Q1: NPN BJT
Q1 3 2 2 QNPN

* Q2: NPN BJT
Q2 2 8 6 QNPN

.model QNPN NPN(IS=1e-14 BF=100)

.end