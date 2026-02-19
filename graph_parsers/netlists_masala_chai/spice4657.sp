spice
* SPICE netlist for the circuit

V1 6 5 DC 0
L1 5 6 Lp
D1 4 2 D
D2 2 5 D

* Assuming ideal diodes
.model D D

* End of netlist