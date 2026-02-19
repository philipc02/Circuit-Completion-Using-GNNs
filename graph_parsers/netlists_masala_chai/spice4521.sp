* SPICE Netlist for the given circuit
* Nodes: 
* 1: v_i
* 2: Common node between R1, R2, and A1
* 3: Common node between R3, R4, A2, and R_L
* 4: Node for positive terminal of A1
* 5: Node connecting bottom side of R3
* 6: Positive supply (ground in labeled node)
* 7: Input node v_i
* 8: Node connecting R_L and output v_O2

* Input Voltage
V1 1 0 DC 0

* Resistors
R1 7 2 1k
R2 2 4 1k
R3 7 5 1k
R4 3 3 1k
RL 3 8 1k

* Operational Amplifiers (Ideal)
* A1 is connected with input from Node 2 and output to Node 4
A1 2 0 4
* A2 is connected with input from Node 3 and output to Node 8
A2 3 0 8

.END