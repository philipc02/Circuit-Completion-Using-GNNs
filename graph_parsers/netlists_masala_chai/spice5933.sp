plaintext
* Example SPICE netlist

M1 3 2 5 5 PMOS
M2 4 2 5 5 NMOS
M3 6 3 4 4 PMOS
M4 3 2 5 5 NMOS
IREF 0 3 DC I_REF_value

VDD 5 0 DC VDD_value
Rout 6 7 ROUT_value

* Analysis
.TRAN 1n 100n
.END