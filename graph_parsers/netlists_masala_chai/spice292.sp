spice
* SPICE Netlist for the Given Circuit

VDD 7 0 DC VDD
VSS 5 0 DC VSS

* Current Source
I1 3 5 DC Iss

* Resistors
Rs1 1 9 Rs
Rs2 6 3 Rs
RL1 8 4 RL
RL2 7 6 RL

* NMOS Transistors
M1 4 9 3 3 NMOS  ; Left NMOS
M2 6 3 3 3 NMOS  ; Right NMOS

* Voltage Probes
Vinput 1 0 DC 0

* Connections
Vin 9 1 DC Vi

.end