spice
* SPICE Netlist
Vgs6 5 10 DC 0
Vn4 4 1 DC 0
Vn2 2 9 DC 0
Vx 3 33 DC 0

Rr2 9 2 r_r2
Ro2 2 8 r_o2
Ro4 7 4 r_o4
Ro6 3 6 r_o6
Rr4 11 1 r_r4

Gm6 6 3 VALUE {g_m6 * (V(5) - V(10))}
Gm4 6 7 VALUE {g_m4 * (V(4) - V(1))}
Gm2 8 2 VALUE {g_m2 * (V(2) - V(9))}

.END