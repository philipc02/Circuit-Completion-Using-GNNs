spice
* NMOS Differential Pair Example
M1 5 8 3 3 NMOS
M2 5 2 3 3 NMOS
RD 4 6 1k
RD 4 6 1k
RS 3 0 10k
V1 8 0 DC
V2 2 0 DC
V+ 6 0 DC 5V
V- 3 0 DC -5V

.model NMOS NMOS (LEVEL=1)

*.end