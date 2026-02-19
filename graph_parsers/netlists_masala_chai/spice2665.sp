spice
* Example SPICE netlist

VDD 6 0 DC VDD_value
VIN 1 0 DC Vin_value

M1 5 1 0 0 NTYPE
M2 2 4 5 5 NTYPE
M3 5 3 6 6 PTYPE
M4 2 4 6 6 PTYPE

I1 3 0 DC I1_value
ISS 5 0 DC ISS_value

* Additional model parameters and simulation commands go here.