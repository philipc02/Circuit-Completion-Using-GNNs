plaintext
* SPICE netlist for the given circuit

V1 1 0 DC V_i
RT 1 3 RT_value
RS 3 5 RS_value
Gm 0 2 VALUE = {g_m * (1 + eta) * V(5)}
RD 4 7 RD_value
RL 4 6 RL_value

* Nodes
* 1 - Positive terminal of V_i
* 0 - Ground
* 3 - Node for I_in, S
* 5 - Node for V_gs
* 7 - Node for V_out
* 6 - Output load node (V_out to ground)

.end