spice
* Define voltage sources
V1 6 0 DC 5V
V2 2 0 DC -5V

* Define resistors
R1 6 7 3R
R2 0 7 R*(1+delta)
RD1 4 8 RD1
RD2 2 5 RD2
RS1 2 6 RS1
RS2 8 5 RS2

* Define transistors
M1 6 7 2 2 NMOS
M2 5 3 4 4 PMOS

* Define nodes for clarity
* 1 - vI
* 2 - V- 
* 3 - vO
* 4 - V+