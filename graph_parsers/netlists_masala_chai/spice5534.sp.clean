* SPICE Netlist for the Amplifier Circuit

V1 2 0 DC <value_of_vs> ; Voltage source v_s
RS 2 4 <value_of_Rs> ; Resistor R_s
GM1 8 3 VALUE={g_m*(V(4,8))} ; Dependent current source g_m*v_{be}
RP 4 8 <value_of_rpi> ; Resistor r_{\pi}
RO 8 3 <value_of_ro> ; Resistor r_o
RL 3 0 <value_of_RL> ; Resistor R_L

* Ground node
R0 9 0 0.0001 ; Tie E to ground with a negligible resistance

.END