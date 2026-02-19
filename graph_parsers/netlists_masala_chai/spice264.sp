spice
* SPICE Netlist

Vg 8 0 DC a_21
I_o1 25 6 DC I_o1
R25 6 0 25
Rup2 3 5 5Meg
Rin3 2 26 Rin3

* NMOS Transistors
M_Q25 25 6 0 0 NMOS
M_Q26 26 6 0 0 NMOS
M_Q27 2 7 0 0 NMOS
M_Q28 33 7 0 0 NMOS

* Connections
* Q26: Drain at net 26, Gate at net 6, Source at net 0
* Q27: Drain at net 2, Gate at net 7, Source at net 0
* Q28: Drain at net 33, Gate at net 7, Source at net 0

.END