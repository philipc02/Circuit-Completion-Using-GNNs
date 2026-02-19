* MOSFET Definitions
M1 2 6 5 5 NMOS
M2 2 3 4 4 PMOS

* Voltage Sources
VDD 4 0 DC VDD_value
Vb 3 0 DC Vb_value
Vin 6 0 DC Vin_value

* .model statements for NMOS and PMOS
.model NMOS NMOS
.model PMOS PMOS

* End of netlist
.end