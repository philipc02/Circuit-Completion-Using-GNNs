* SPICE netlist for the given schematic

* Voltage Sources
V1 9 0 DC VG

* Current Sources
I1 3 9 DC IS1
I2 7 9 DC IS2

* NMOS Transistors
M1 3 5 8 8 NMOS
M2 4 6 8 8 NMOS
M3 3 2 8 8 NMOS

* PMOS Transistors
M4 2 2 3 3 PMOS
M5 7 2 3 3 PMOS

* Buffer Stage
M6 2 2 2 2 NMOS
M7 7 2 2 2 PMOS

* Output Transistor
M8 6 2 2 2 PMOS

* Resistors
R1 2 5 R1_VALUE
R2 4 2 R2_VALUE

* .MODEL statements for the transistors
.MODEL NMOS NMOS (LEVEL=1)
.MODEL PMOS PMOS (LEVEL=1)

* End of netlist