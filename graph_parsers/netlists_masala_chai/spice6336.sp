spice
* SPICE Netlist for the given schematic

* PMOS Transistors
M_Q2 4 4 9 9 PMOS
M_Q4 6 6 9 9 PMOS
M_Q6 3 3 7 7 PMOS
M_Q8 2 2 8 8 PMOS

* NMOS Transistors
M_Q1 3 7 5 5 NMOS
M_Q3 7 6 5 5 NMOS
M_Q5 5 2 1 1 NMOS
M_Q7 8 6 1 1 NMOS

* Voltage Source
VDD 9 0 DC VDD

* Nodes
* Node annotations in the second image:
* 1 = Ground
* 2 = R
* 3 = Q
* 4 = Q̅
* 5 = S
* 6 = Φ
* 7 = Connected node between Q1 and Q3
* 8 = Source of Q7
* 9 = VDD