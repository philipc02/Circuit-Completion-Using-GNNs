plaintext
* SPICE Netlist for given circuit

Vsig 5 0 DC 0
Rsig 5 3 Rvalue

M1 2 3 0 0 NMOS L=1u W=1u
RL 2 4 RLvalue

* .model NMOS NMOSLEVEL=1 PARAMETERS_GO_HERE

.end