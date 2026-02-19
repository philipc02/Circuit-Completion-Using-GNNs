* SPICE Netlist
* Components:
* Transistors
M1 3 8 4 4 PMOS
M2 3 3 4 4 PMOS
M3 4 6 24 24 NMOS
M4 4 4 24 24 NMOS
M6 5 4 24 24 NMOS

* Current Sources
I1 23 3 ITAIL
I2 3 5 I7

* Capacitors
C1 3 0 C1
Cgd1 3 8 Cgd1
Cgs1 8 4 Cgs1
Cc 4 5 Cc
Cs 6 0 Cs

* Voltage Source
Vss 24 0 DC 0
Vdd 3 0 DC Vdd
Vi 9 0 DC Vi

.ends