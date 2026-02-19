spice
* SPICE Netlist
* Components

* Current Source
I1 6 0 DC IQ

* PMOS Transistors
M1 6 4 5 5 PMOS
M2 6 2 2 2 PMOS

* NMOS Transistors
M3 5 3 3 3 NMOS
M4 2 3 3 3 NMOS

* Voltage Sources
V1 V+ 0 DC 10
V2 V- 0 DC -10

* Nodes
* v1 -> node 4
* v2 -> node 2
* vo -> node 2

.END