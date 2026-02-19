spice
* Differential Amplifier SPICE Netlist

* Voltage Sources
Vdd 3 0 DC VDD
Vcont 1 0 DC Vcont
Vin 2 0 DC Vin

* Current Source
Iss 6 0 DC Iss

* Capacitors
CL1 5 0 CL
CL2 5 0 CL

* NMOS Transistors
M1 5 1 6 6 NMOS
M2 5 2 6 6 NMOS

* PMOS Transistors
M3 4 1 3 3 PMOS
M4 5 2 3 3 PMOS

.end