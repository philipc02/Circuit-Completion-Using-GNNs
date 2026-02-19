plaintext
* SPICE netlist for the given schematic

* NMOS Transistors
M1 3 3 2 2 NMOS
M2 2 3 2 2 NMOS

* PMOS Transistors
M3 2 4 4 4 PMOS
M4 4 4 4 4 PMOS
M5 4 4 4 4 PMOS

* Resistor
R1 2 6 R

* Current Sources
I_REF 2 6 DC IREF
I2 2 0 DC I2
I3 4 0 DC I3
I4 4 0 DC I4
I5 4 0 DC I5

* Voltage Sources
V1 3 0 DC 1.0
V2 4 0 DC -1.0

* Model Definitions
.model NMOS NMOS
.model PMOS PMOS

.end