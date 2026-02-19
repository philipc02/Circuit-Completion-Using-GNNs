* SPICE netlist for the circuit

* Voltage Sources
VDD 6 3 DC
VIN 7 0 DC
VCONT 1 3 DC
VP_CONT 8 0 DC

* Resistors
R1 6 5 R
R2 5 3 R

* PMOS Transistors (Drain Gate Source)
M1 5 0 8 8 PMOS
M2 5 3 8 8 PMOS

* NMOS Transistors (Drain Gate Source)
M3 5 4 2 2 NMOS
M4 3 5 2 2 NMOS
M5 2 1 7 7 NMOS
M6 1 2 7 7 NMOS

.END