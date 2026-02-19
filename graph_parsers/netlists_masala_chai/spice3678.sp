spice
* SPICE Netlist for the schematic

V1 4 0 DC 5V
V2 5 0 DC -5V

RG 2 1 50k
RD 4 3 10k
RS 6 5 10k

M1 3 2 6 6 NMOSmodel

.model NMOSmodel NMOS