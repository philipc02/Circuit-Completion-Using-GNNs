plaintext
* SPICE Netlist
M2 2 4 3 3 PMOS
CL 2 0
VDD 3 0 DC 5V
Vin 4 0 DC 0
Vout 2 0

* PMOS Model
.model PMOS PMOS (LEVEL=1 VTO=-1 KP=20u)