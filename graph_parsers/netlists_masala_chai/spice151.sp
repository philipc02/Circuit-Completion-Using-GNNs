* SPICE Netlist

* Voltage source
VDD 6 0

* Current source
IIN 6 5 DC 0

* Resistors
R1 5 4 R1_value
R2 3 2 R2_value

* NMOS Transistors
M1 4 4 7 7 NMOS_MODEL
M2 3 4 2 2 NMOS_MODEL

* .MODEL statement
.model NMOS_MODEL NMOS

* Simulation commands
*.op
*.end