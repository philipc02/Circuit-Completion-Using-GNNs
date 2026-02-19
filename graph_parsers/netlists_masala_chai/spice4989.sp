plaintext
*SPICE netlist for the given schematic

V1 7 0 DC Vin
V2 5 6 DC VGate
V3 2 0 DC VEN

M1 6 2 0 0 NMOS
M2 3 6 7 7 PMOS

R1 6 3 R1_value
RL 3 4 RL_value

* Voltage source and transistor models
.model NMOS nmos
.model PMOS pmos

.end