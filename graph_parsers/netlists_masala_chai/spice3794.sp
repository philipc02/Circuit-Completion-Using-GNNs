plaintext
* SPICE netlist for the given schematic

V1 3 0 DC Vi
C_C1 3 5 CC1
R_G 5 0 1MEG
M1 4 5 6 6 NMOS
R_S 6 0 Rs
R_D 7 4 Rd
C_C2 4 2 CC2
R_L 2 0 Rl
V_DD 7 0 DC Vdd

* End of netlist