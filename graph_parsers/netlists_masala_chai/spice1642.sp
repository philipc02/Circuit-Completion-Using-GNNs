plaintext
* SPICE netlist for the given circuit

VDD 5 0 DC 5V
VIN 3 0 DC V_in

RD 5 7 RD_VALUE
RI 6 8 RI_VALUE
CI 7 8 CI_VALUE

M1 5 8 2 2 NMOS_MODEL
M2 7 3 0 0 NMOS_MODEL

I1 2 4 DC I1_VALUE

* End of netlist

.model NMOS_MODEL NMOS LEVEL=1