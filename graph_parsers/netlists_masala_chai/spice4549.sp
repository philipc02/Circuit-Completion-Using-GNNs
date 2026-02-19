spice
* SPICE Netlist for the given schematic

* PMOS Transistor M_L
M1 v_o 2 v_dd v_dd PMOS

* NMOS Transistor M_DA
M2 2 A 0 0 NMOS

* NMOS Transistor M_DB
M3 2 B 0 0 NMOS

* Voltage Source
V1 v_dd 0 DC 5V

* Model Definitions
.model PMOS PMOS
.model NMOS NMOS

.end