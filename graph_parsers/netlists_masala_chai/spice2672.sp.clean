spice
* Transistor netlist
M1 2 Vin1 4 4 NMOS
M2 3 Vin2 4 4 NMOS
M3 2 F VDD VDD PMOS
M4 3 F VDD VDD PMOS
M5 4 Vb 0 0 NMOS

* Voltage supply
VDD VDD 0 DC 5

* Define inputs and bias
VIN1 Vin1 0 DC 1.2
VIN2 Vin2 0 DC 1.2
VB Vb 0 DC 1.5

* Output
VOUT Vout 3

* .model definitions (typically needed for actual simulation)
.model NMOS NMOS (level=1)
.model PMOS PMOS (level=1)

.end