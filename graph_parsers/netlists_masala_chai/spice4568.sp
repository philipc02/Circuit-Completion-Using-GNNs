spice
* NMOS model
.model NMOS NMOS(Level=1 VTO=0.7)

* PMOS model
.model PMOS PMOS(Level=1 VTO=-0.7)

* Voltage source
V1 6 0 DC VDD

* PMOS transistors
M1 7 2 6 6 PMOS
M2 7 4 6 6 PMOS

* NMOS transistors
M3 3 2 7 7 NMOS
M4 5 4 3 3 NMOS
M5 5 2 3 3 NMOS
M6 F 4 5 5 NMOS

* End of netlist
.END