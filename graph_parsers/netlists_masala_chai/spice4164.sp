spice
* SPICE Netlist for the given circuit

* Voltage Sources
V1 6 0 DC 10V
V2 3 0 DC -10V

* NMOS Transistor M_n
M1 6 5 6 6 NMOS_3T

* PMOS Transistor M_p
M2 6 2 3 3 PMOS_3T

* Resistor
RL 4 0 1k

* Input Connection (vi)
Vin 2 0 DC 0

* Output Connection (vo)
Vout 4 0 DC 0

.model NMOS_3T NMOS
.model PMOS_3T PMOS

.end