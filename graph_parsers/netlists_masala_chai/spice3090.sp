* Transistor Definitions
M1 4 5 6 6 NMOS
M2 4 5 6 6 NMOS
M3 2 1 7 7 PMOS
M4 2 3 7 7 PMOS
M5 4 1 2 2 PMOS
M6 4 3 2 2 PMOS
M7 4 6 5 5 NMOS
M8 4 6 5 5 NMOS

* Current Source
I1 5 0 DC 0.5mA

* Voltage Source
VDD 7 0 DC 5V

* Simulation Commands
.tran 1n 10n
.end