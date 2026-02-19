spice
* SPICE Netlist

Vsig 4 0 DC 0
VDD 6 0 DC VDD_value

Rsig 4 2 Rsig_value
RL 3 0 RL_value
CL 3 0 CL_value

M1 3 2 5 5 NMOS_MODEL

* Model Definitions
.model NMOS_MODEL NMOS

* End