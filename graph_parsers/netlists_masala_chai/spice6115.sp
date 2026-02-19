spice
* SPICE netlist for the given circuit

Vsig 8 0 DC 0
Rsig 8 3 10k
Rx 3 2 1k
RL 9 0 10k
Cmu 2 5 10p
Cpi 2 0 10p
CL 6 0 100p

G1 5 0 2 0 gm
Ro 5 6 10k

.END