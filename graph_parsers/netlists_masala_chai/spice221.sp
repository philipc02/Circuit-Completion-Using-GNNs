spice
* SPICE Netlist

V1 6 0 DC Vs

* Transistors
M1 6 5 5 5 PMOS
M2 5 5 5 5 PMOS
M3 3 2 2 2 NMOS
M4 4 2 2 2 NMOS
M5 2 3 2 2 NMOS

* Capacitors
C1 5 3 C1_value
C2 2 4 C2_value
Cp 3 2 Cp_value

* AC Analysis
.ac dec 100 1 1MEG

.end