spice
* SPICE Netlist

* Define Components
I1 8 9 DC 1A
R1 6 10 1k
D1 2 5 Dmodel
VB 3 5 DC 1V

* Define Models
.model Dmodel D

* Define Nodes
V1 7 4 DC 0