spice
* SPICE Netlist for the given circuit

R_rb 6 2 r_b
C_Cpi 2 3 Cpi/(1+g_m*R_L)
R_rpi 3 5 (1+g_m*R_L)*r_pi
R_RL 3 4 R_L

* Ground definition
V_GND 4 0 DC 0

.end