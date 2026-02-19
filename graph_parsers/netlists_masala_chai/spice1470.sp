plaintext
* Netlist
M_Q1 net_Q1_d V_in 0 0 NMOS
M_Q2 net_Q2_d V_b1 net_Q1_d net_Q1_d NMOS

I_I1 net_Q2_d 3 DC I1_VALUE
V_VCC 3 0 DC VCC_VALUE
V_Vb1 4 0 DC Vb1_VALUE
V_Vin V_in 0 DC Vin_VALUE

* Connections
* net_Q1_d corresponds to the internal node between Q1 drain and Q2 source
* net_Q2_d corresponds to the node leading to V_out
* 3 is VCC and current source node
* 4 is for V_b1