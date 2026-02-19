* SPICE Netlist
V1 5 0 DC
RS 5 2 50k
RL 2 4 1k
* Ideal Op-Amp Model
* Edevice_name <+ terminal> < - terminal> < + terminal> < - terminal> gain
E1 3 0 2 0 1e6
* Vout is at node 4