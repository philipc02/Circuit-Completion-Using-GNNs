spice
* SPICE Netlist for the given schematic

V1 8 6 DC Vi

Rif 8 5 Rif
RB 5 2 RB
rpi 5 2 rpi
RE 2 9 RE
Rof 2 7 Rof
RC 7 6 RC
RL 7 6 RL

I0 6 7 DC Io

Gm 0 2 Value = {gm * V(5,2)}

* Ground Node
0 9 0