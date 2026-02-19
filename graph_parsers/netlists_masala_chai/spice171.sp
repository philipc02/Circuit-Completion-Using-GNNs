spice
* SPICE Netlist for the given schematic

VDD 4 5 DC VDD
VGS 5 5 DC VGS

I_D1 2 5 DC ID1
I_D2 3 4 DC ID2

M1 8 5 5 5 NMOS_MODEL
M2 3 2 4 4 PMOS_MODEL

* Models (provide appropriate model parameters)
.model NMOS_MODEL NMOS
.model PMOS_MODEL PMOS

.end