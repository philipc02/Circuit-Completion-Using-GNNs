plaintext
* SPICE netlist for the circuit

R_rpi 4 6 rpi
I_i1 3 5 DC <value>    * Specify the current value
G_gmvpi 3 6 VALUE={gm*(V(4,6))}    * Voltage-controlled current source
R_RL 2 0 RL
V_vx 2 0 vx