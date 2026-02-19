* SPICE Netlist for the given circuit

V1 1 0 DC 5V
R1 1 2 20k
D1 2 0 D
I1 2 3 DC I
V0 2 3 DC 0

.model D D

.end