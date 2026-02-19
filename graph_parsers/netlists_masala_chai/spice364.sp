spice
* SPICE netlist for given schematic
R_F 5 2 RF
R_E 4 2 RE
R_L 3 0 RL
V_x 3 0 Vx DC 0
I_L 6 0 IL DC 0

* Ideal Op-Amp
* Non-inverting input: 3
* Inverting input: 2
* Output: 6
Ea 6 0 3 2 1e6

.END