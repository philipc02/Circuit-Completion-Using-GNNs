spice
* SPICE netlist for the given schematic

VCC 1 0 DC 5V
VEE 5 0 DC -5V
VS 6 0 AC 1V

RE 4 5 3.3k
RO 2 3 RO
RL 2 3 1k

CC 4 2 CC

RIB 6 5 RIB

Q1 4 5 3 NPN

.END