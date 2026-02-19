spice
* Spice netlist for the given circuit

V1 8 7 DC 0
RS 8 2 200
C1 2 6 4.7u
R1 6 0 1.2k
V2 6 0 DC 5
V3 7 0 DC -5
C2 2 4 1u
RD 4 0 1.2k
RL 4 5 50k

* Input and output
* vi between nodes 8 and 7
* vo at node 5