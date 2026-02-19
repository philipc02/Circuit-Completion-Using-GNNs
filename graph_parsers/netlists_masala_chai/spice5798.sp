spice
* NMOS Amplifier Circuit

M1 4 2 3 3 NMOS_MODEL
VGS 2 3 DC Vgs_value
RD 6 4 RD_value

* DC source for drain terminal (VDD)
VDD 6 0 DC VDD_value

* .model statement for NMOS
.model NMOS_MODEL NMOS (LEVEL=1)

* End of netlist
.end