* SPICE Netlist

* Voltage Source
V1 2 0 DC (Vid/2)

* Current Source
I1 6 2 DC (gm1*(Vid/2))

* PMOS Transistor
M1 5 2 6 6 PMOS

* NMOS Transistor
M7 2 2 3 3 NMOS

* Resistors
Rro1 5 2 ro1
Rro7 2 4 ro7
RR7 3 0 R7

.end