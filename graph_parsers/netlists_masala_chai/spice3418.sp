plaintext
* SPICE netlist for the given schematic
VDD 4 0 DC VDD
VT 6 0 DC VT
RT 6 7 R_T
C1 7 2 C_1
RG1 4 2 R_G1
RG2 2 8 R_G2
RS 8 0 R_S
RD 4 2 R_D
CO 2 3 C_0
RL 3 0 R_L
CS 2 5 C_S

* NMOS transistor
M1 2 2 8 8 NMOS

* End of netlist