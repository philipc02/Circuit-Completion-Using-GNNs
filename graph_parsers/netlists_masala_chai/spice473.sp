* SPICE netlist for the given schematic

Rb 10 9 r_b
Ib 9 7 DC I_b
Rpi 7 5 r_pi
Cpi 5 6 C_pi
Gmvpi 6 4 VALUE={g_m * V(6,5)}
Ro 4 8 r_o
Ic 11 3 DC I_c

* Node connections: 
* Node 2 is the output node i_o