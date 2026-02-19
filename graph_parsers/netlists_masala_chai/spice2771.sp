* SPICE Netlist
VDD 3 0 DC VDD_VALUE
Vin 1 0 DC VIN_VALUE

RS 2 4 RS_VALUE
CGS2 2 5 CGS2_VALUE
CL 5 0 CL_VALUE

M2 6 2 3 3 PMOS_MODEL
M1 6 1 5 5 NMOS_MODEL

* Model definitions
.model PMOS_MODEL PMOS
.model NMOS_MODEL NMOS

* Voltage, Resistance, and Capacitance Values
.param VDD_VALUE = 5V
.param VIN_VALUE = 1V
.param RS_VALUE = 1k
.param CGS2_VALUE = 1p
.param CL_VALUE = 1p