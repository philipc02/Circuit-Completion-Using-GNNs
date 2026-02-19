plaintext
* PMOS Transistor Q_P
M1 6 6 3 3 PMOS

* NMOS Transistor Q_N
M2 6 4 5 5 NMOS

* VDD Source
VDD 3 0 DC VDD

* Input Voltage Source
VI 4 0 DC VI

* Model Definitions
.model PMOS PMOS (level=1)
.model NMOS NMOS (level=1)

* End of Netlist
.end