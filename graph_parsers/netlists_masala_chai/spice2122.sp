spice
* Example SPICE netlist for the circuit

VDD VDD 0 DC VDD_VALUE
Vin Vin 0 DC VIN_VALUE

RD VDD 2 RD_VALUE

M1 2 Vin 3 3 NMOS_MODEL
M2 3 3 3 3 NMOS_MODEL

*.model NMOS_MODEL NMOS (level=1 VTO=...)
*.param VDD_VALUE=VALUE
*.param VIN_VALUE=VALUE
*.param RD_VALUE=VALUE

.end