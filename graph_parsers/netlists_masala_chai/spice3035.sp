spice
* Components
M1 5 7 3 3 NMOS
M2 9 7 3 3 NMOS
M3 5 2 22 22 PMOS
M4 2 2 22 22 PMOS
M5 2 2 3 3 NMOS
M6 2 2 3 3 NMOS
M7 2 3 24 24 NMOS
M8 2 3 24 24 NMOS
M9 22 2 VDD VDD PMOS
M10 2 2 VDD VDD PMOS

R1 8 0 <value>
R2 3 0 <value>
R3 5 8 <value>
R4 5 2 <value>

I1 7 0 <current_value>

* Voltage supply
VDD VDD 0 <voltage_value>

* Ensure to define MOSFET models
.model NMOS NMOS (level=1)
.model PMOS PMOS (level=1)

.end