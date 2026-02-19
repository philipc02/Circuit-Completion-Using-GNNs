plaintext
* SPICE netlist

* Voltage Sources
VDD 6 0 DC VDD
VIN 5 0 DC VIN

* Current Source
I1 3 0 DC I1

* Resistors
RD 6 4 RD
RS 5 3 RS

* NMOS Transistor
M1 4 2 3 3 NMOS

* Operating points and analysis commands can be added here if needed
*.OP
*.DC VIN START STOP INCREMENT
*.TRAN START END INCREMENT

.end