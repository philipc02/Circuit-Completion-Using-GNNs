spice
* SPICE Netlist
VDD 8 0 DC V_DD

I1 8 6 DC 1mA

M1 3 P 5 5 NMOS_model
M2 2 P 6 6 NMOS_model
M3 8 3 7 7 PMOS_model
M4 2 3 9 9 PMOS_model
M5 5 2 2 2 NMOS_model
M6 6 2 2 2 NMOS_model

.model NMOS_model NMOS (level=1)
.model PMOS_model PMOS (level=1)

.end