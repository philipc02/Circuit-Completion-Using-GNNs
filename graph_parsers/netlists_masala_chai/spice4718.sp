spice
* SPICE netlist for the given schematic

VBB 2 0 DC 10
VCC 5 0 DC 10

RB 2 3 1MEG
RC 3 5 2K

Q1 3 2 0 NPN

.MODEL NPN NPN
.END