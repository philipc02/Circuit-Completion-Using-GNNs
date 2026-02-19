plaintext
* SPICE Netlist for the given schematic

* NMOS Transistors
M_Q1 2 5 1 1 NMOS
M_Q2 3 5 2 2 NMOS
M_Q3 4 5 5 5 NMOS
M_Q4 5 5 6 6 NMOS

* Resistors
R1 3 6 R_value
R2 5 6 R_value

* Current Sources
I_Q1 1 8 DC -10
I_Q2 5 7 DC -10

* Voltage Sources
Vdd 6 0 DC +10

* Define NMOS model
.model NMOS NMOS

.end