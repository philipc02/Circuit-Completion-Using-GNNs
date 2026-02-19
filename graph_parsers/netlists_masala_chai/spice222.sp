plaintext
* SPICE Netlist for the given circuit
M1 3 1 4 4 PMOS
M2 3 2 4 4 PMOS
M3 5 1 6 6 NMOS
M4 5 2 6 6 NMOS
C1 3 2 5pF
CS 3 5 1pF
CL 2 0 2pF
Vs 6 0 DC <value>
* Define .model for PMOS and NMOS transistors
.model PMOS PMOS
.model NMOS NMOS

*.end