spice
* SPICE Netlist
V1 9 2 DC [insert value]
RT 9 6 [insert value]
RG 6 2 [insert value]
V_P1 2 3 DC [insert value]
I_BN1 4 3 DC (βn1 * V(V_P1))
V_GM2 8 3 DC (g_m2 * V(V_X2))
I_GM2 7 5 DC (g_m2 * V(V_X2))
RD 8 5 [insert value]