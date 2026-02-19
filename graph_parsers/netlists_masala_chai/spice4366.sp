spice
* SPICE Netlist
VPlus 4 0 DC 5
VMinus 8 0 DC -5

RS 4 5 2k
RD1 3 8 1k
RD2 3 8 1k

M1 7 9 3 3 PMOS
M2 5 6 3 3 PMOS

Vin1 9 0 DC 0
Vin2 2 0 DC 0

* Define models
.model PMOS PMOS(Level=1)

.END