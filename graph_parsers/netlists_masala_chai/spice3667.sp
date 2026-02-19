* SPICE Netlist for the given circuit

R1 1 2 10k
R2 1 3 10k
R3 1 4 10k
R4 4 3 10k
D1 2 4 D

V1 1 0 DC 0
V2 4 0 DC 0

* .model declaration for the diode
.model D D

* End of netlist