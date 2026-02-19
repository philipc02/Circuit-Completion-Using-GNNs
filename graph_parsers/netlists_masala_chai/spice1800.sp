plaintext
* SPICE Netlist for the given schematic
.model NMOS NMOS level=1
.model PMOS PMOS level=1

Mn1 2 5 6 6 NMOS
Mp2 2 3 8 8 PMOS

Iin 2 7 DC 0
Vb 5 0 DC Vb_value
Vdd 8 0 DC Vdd_value

Rd 3 4 RD_value
Rf 6 5 RF_value

* Laser connection
Rlaser 2 9 Rlaser_value

.tran 1ns 100ns
.end