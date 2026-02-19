spice
* SPICE Netlist
* NMOS: M<name> <drain> <gate> <source> <body> <model> <W> <L>
* PMOS: M<name> <drain> <gate> <source> <body> <model> <W> <L>

M1 4 1 3 3 NMOS
M2 5 2 3 3 NMOS
M3 2 4 5 5 PMOS
M4 6 4 5 5 PMOS

* Current Source
I1 3 0 DC Iss

* Capacitor
C1 4 5 C1

* Voltage Supply
VDD 6 0 DC VDD
V1 1 0 DC 1.5

.model NMOS NMOS
.model PMOS PMOS

.END