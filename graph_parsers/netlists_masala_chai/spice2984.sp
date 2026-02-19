* SPICE Netlist

* Voltage Sources
Vt 3 0 DC VALUE

* Resistors
RD1 8 5 VALUE
RD2 4 5 VALUE
R1 6 3 VALUE
R2 8 6 VALUE

* NMOS Transistors
M1 8 6 2 2 NMOS_MODEL
M2 4 7 2 2 NMOS_MODEL

* Voltage Supply
VDD 5 0 VALUE

* Output
Vout 4

* Models
.model NMOS_MODEL NMOS (Level=1)