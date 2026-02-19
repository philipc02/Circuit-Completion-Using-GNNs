spice
* SPICE Netlist for the given schematic

* Voltage Source
V1 2 7 DC 0

* Current Sources
I_beta_n1_Vc 6 7 DC 0
I_beta_n2_Vc 4 3 DC 0
I_beta_o_Vt2 8 9 DC 0

* Resistors
R_RT 5 2 1k
R_RE1 6 5 1k
R_RB1 3 5 1k
R_RE2 3 4 1k
R_RB2 9 3 1k

* Voltage Probes
V_in 6 0 DC 0
V_out 9 0 DC 0

.END