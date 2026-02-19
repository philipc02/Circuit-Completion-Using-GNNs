spice
* SPICE netlist for the given schematic

Vi 8 0 DC

RS 8 5 1k

Gm1 5 0 3 5 gm1
Gm2 3 0 3 6 gm2
Gm3 3 0 3 2 gm3

RD 4 0 RD_value
R1 6 0 10k
R2 2 6 40k
RL 2 0 4k

.END