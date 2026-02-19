spice
* Netlist for the given circuit

* Voltage Source
V1 2 5 DC 0

* Current Source
I1 2 0 DC i

* Resistors
R1 6 5 RS_plus_rb
R2 2 6 r_pi
R3 3 5 RL

* Dependent Current Source
G1 2 4 3 0 gm

* Node Identification:
* 1 - Ground (0)
* 2 - Top left node
* 3 - Right top node
* 4 - Point joining gm control
* 5 - Bottom shared node (common ground)
* 6 - Left node, top of RS + rb

.end