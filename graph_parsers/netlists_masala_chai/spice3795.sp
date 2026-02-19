spice
* SPICE Netlist for the given circuit

Vi 5 0 DC 0

RS 6 4
RG 5 0 100k
RD 3 7
RL 2 0

CC1 5 0
CC2 3 4
CS 4 0

M1 4 5 3 3 NMOS

VCC 6 0 DC 5V
VEE 7 0 DC -5V

* End of netlist