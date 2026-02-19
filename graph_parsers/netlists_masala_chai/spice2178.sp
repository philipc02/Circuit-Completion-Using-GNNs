spice
* SPICE Netlist
V1 2 0 DC Vin
G1 3 0 2 0 1/gm
R1 3 0 RS_RO

.model RS_RO RES(RS=RS || RO)
.END