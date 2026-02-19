spice
* SPICE netlist
* NMOS: M<name> <drain> <gate> <source> <source>
* PMOS: M<name> <drain> <gate> <source> <source>

M1 3 2 5 5 NMOS
M2 7 2 3 3 NMOS
M3 3 4 6 6 PMOS
M4 2 2 6 6 PMOS
M5 7 2 3 3 NMOS
M6 7 4 6 6 PMOS

* Current Sources
ID1 6 4 DC (5/1)
ID2 6 2 DC (5/1)
I01 7 3 DC
I02 7 6 DC

* Resistor
R 2 3 R

*.END