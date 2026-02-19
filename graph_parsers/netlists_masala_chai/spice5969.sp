* NMOS Transistor Q1
M1 3 5 3 3 NMOS

* NMOS Transistor Q2
M2 6 5 2 2 NMOS

* Current Source
I1 1 3 100uA

* Resistors
R1 3 3 Rs
R2 2 2 Rs

* Nodes
* 1: Current Source positive terminal
* 2: Resistor Rs (right side) connection to Q2 source and ground
* 3: Resistor Rs (left side) connection to Q1 source and ground
* 5: Common gate node for Q1 and Q2
* 6: Drain of Q2 connected to R_out

.end