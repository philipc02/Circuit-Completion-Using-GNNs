plaintext
* SPICE Netlist for the given schematic

MN1 1 2 0 0 NMOS
MN2 3 2 0 0 NMOS
MP1 1 3 2 2 PMOS
MP2 3 3 2 2 PMOS

R1 1 4 R_value
R2 4 3 R_value

Vout1 1 0 DC 0
Vout2 3 0 DC 0
Vout_CM 4 0 DC 0

.model NMOS NMOS (Level=1)
.model PMOS PMOS (Level=1)

* End of netlist