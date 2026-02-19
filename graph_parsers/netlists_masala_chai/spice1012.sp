spice
* SPICE netlist for the given schematic

VCC 3 0 DC 2.5

R1 3 2 5k
RB 2 0 1k ; Assumed value for RB since it was not specified
R2 4 0 1k

Q1 5 2 4 QNPN

.model QNPN NPN (BF=100)

.end