plaintext
* Schematic Netlist

* Voltage and Current Sources
VDD 3 0 DC V_DD
IIN 2 5 DC I_IN

* Transistors
M1 6 2 5 5 NMOS
M2 4 2 5 5 NMOS

* Resistors
RD1 3 6 RD1
RD2 3 4 RD2
RF1 2 5 RF
RF2 2 5 RF

* Simulation Commands
*.DC ...
*.AC ...
*.TRAN ...
.END