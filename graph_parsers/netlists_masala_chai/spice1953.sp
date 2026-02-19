plaintext
* SPICE Netlist

* Voltage Source
V1 2 0 V_I

* Resistors
R_RD 1 3 2rD
R_RL 6 0 RL

* PMOS Transistor
M_Q1 6 2 1 1 PMOS

* NMOS Transistors
M_Q2 6 4 5 5 NMOS
M_Q4 4 Vin 5 5 NMOS

* Connections:
* Node 0 is the ground.
* Node 1 is connected to the drain of Q1 and the source of Q1 is connected to V1 (node 2).
* Node 3 is an intermediate node connected to the resistor 2rD.
* Node 4 is the gate of Q2 and the drain of Q4.
* Node 5 is the common source for NMOS transistors (Q2 and Q4).
* Node 6 is the drain of Q2 and connected to the output Vout.