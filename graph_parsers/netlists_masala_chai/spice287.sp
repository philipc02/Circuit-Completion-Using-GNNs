spice
* SPICE Netlist
Vi 7 11 DC 0
RS 7 9  R_S
Rx 9 3  r_x
Rin 3 12  r_in
Cin 3 5  C_in
CM 8 6  C_M
Gm 10 6 VALUE = {g_m * V(12, 6)}
RL 10 4  R_L
Rin2 5 11 1G

* .END