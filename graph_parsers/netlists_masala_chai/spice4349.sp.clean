* SPICE Netlist for the given schematic

* Voltage Sources
Vplus 5 0 DC 10V
Vminus 2 0 DC -10V

* Current Source
I1 2 4 DC 2mA

* Resistors
RC1 3 5 2k
RC2 3 5 2k

* NPN Transistors
Q1 2 1 4 QNPN
Q2 3 1 5 QNPN

.model QNPN NPN (IS=1E-14 BF=100)

* Nodes Legend
* 1 = v1
* 2 = v2
* 3 = Common collector node for Q1 and Q2
* 4 = Node for I_Q and Q1/Q2 emitters
* 5 = V+ node