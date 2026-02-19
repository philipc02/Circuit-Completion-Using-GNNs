* SPICE Netlist for Differential Amplifier

* Resistors
R1 3 4 RC
R2 5 4 RC
R3 4 2 RL
R4 2 4 RL

* Current Source
I1 6 2 IBIAS

* Voltage Sources
VCC 5 0 DC VCC
VEE 6 0 DC VEE

* Additional Nodes
* Node 1: vi/2
* Node 2: Ground / v0q
* Node 3: Top connection of RC resistors
* Node 4: Node for RL resistors
* Node 5: Output / +VCC
* Node 6: -VEE

.END