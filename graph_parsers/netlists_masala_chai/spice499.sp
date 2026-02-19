plaintext
* SPICE Netlist

* Elements
I1 7 2 DC IS
RS 5 2 1000
RL 3 6 1000
Q1 4 5 2 NPN

* Nodes
* 1: Ground
* 2: Transistor emitter
* 3: Transistor collector, RL
* 4: RL, Vo(+)
* 5: RS, Transistor base, I1
* 6: Vo(-)
* 7: I1, RS

* Transistor Model
.model NPN NPN

* Simulation Commands
.OP
.end