plaintext
* SPICE netlist for the schematic

* NMOS Transistors
M1 3 2 7 NMOS
M2 6 3 7 NMOS

* PMOS Transistors
M3 2 4 5 PMOS
M4 3 6 5 PMOS

* Voltage Sources
VIN 2 0 DC 0
VCC 4 0 DC 5

* Resistors
R1 2 7 1k
R2 3 2 1k
R3 5 6 1k
R4 6 0 1k
R5 3 5 1k
R6 7 0 1k

* Assign NMOS and PMOS models 
.model NMOS nmos
.model PMOS pmos

.end