* SPICE netlist
V1 1 5 DC 0 * Voltage Source (v_s)
Rvgs 6 2 DC 0 * Voltage (v_gs), 0V just for reference
Igm 3 2 DC 0.001 * Current Source g_m*v_gs
Igmb 3 5 DC 0.001 * Current Source g_mb*v_bs
RS 2 5 RS * Resistor
ro 3 5 r_o * Resistor
RD 3 4 R_D * Resistor
V2 4 0 DC Vo * Voltage source used to measure output voltage

* Notes:
* All voltages in this case are referenced with respect to node 5 (ground/common).
* The DC values for current sources reflect the symbolic representation.