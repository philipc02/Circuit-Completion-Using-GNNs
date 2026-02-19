* SPICE Netlist
Vi 7 4 DC 

Rb 7 3 rb
Rpi 3 5 rpi
RL 6 8 RL

C1 5 3 C1

Gm 2 5 2 gm

Vout 6 2 DC 0

* Node Definitions:
* 1 - not used
* 2 - common node for GmVπ, Vout, C1
* 3 - Rb, Rpi, C1 intersection
* 4 - Negative terminal of Vi
* 5 - Negative terminal of Rpi, positive terminal of C1
* 6 - Positive terminal of Vout, RL
* 7 - Positive terminal of Vi, Rb
* 8 - RL, Vout output