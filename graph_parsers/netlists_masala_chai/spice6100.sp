plaintext
* SPICE netlist

Vsig 6 8 DC 0
Rsig 8 5 100k
Rg 5 7 10k
Rpi 7 4 1k
Ro 3 2 50k
RC 2 2 1k
RL 2 2 1k
Cmu 4 2 1pF
Cpi 4 2 1pF

Gm 3 2 VALUE = {g_m * (V(4,2))}

* Ground
E 5 0 DC 0

* End of netlist