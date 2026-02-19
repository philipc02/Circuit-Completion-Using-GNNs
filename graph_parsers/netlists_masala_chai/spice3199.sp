plaintext
* NMOS Transistors
M_Q1  2 4 3 3 NMOS
M_Q2  2 2 3 3 NMOS
M_Q3  6 5 4 4 NMOS
M_Q4  2 3 4 4 NMOS
M_M9  7 F 5 5 NMOS
M_M11 2 E 5 5 NMOS

* PMOS Transistors
M_M2  1 2 2 2 PMOS
M_M10 1 VDD 3 3 PMOS

* Voltage Source
V_VDD VDD 0 DC 5V

* Resistors
R_R1  3 2 1k
R_R2  2 2 1k
R_R3  2 2 1k
R_R4  F 0 1k
R_R5  E 0 1k
R_R6  F 0 1k

* Operational Amplifier
X_A1 E F Vout OPAMP

* End of Netlist