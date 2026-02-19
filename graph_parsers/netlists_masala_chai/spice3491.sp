plaintext
* SPICE Netlist for the given circuit

VCC 5 0 DC 5V
VEE 7 0 DC -5V
VIN 3 0 AC 1V

Q2 4 3 2 NPN

R2 5 3 1k
R3 3 7 1k
R 4 2 1k
RL 2 0 1k

CC 3 3 1uF

.MODEL NPN NPN(IS=1E-14 BF=100)

.END