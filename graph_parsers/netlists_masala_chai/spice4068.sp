spice
* Circuit Netlist
VDD 2 0 DC 10V
Vi 8 4 DC 0V

R1 2 6 234k
Ri 4 6 10k
R2 6 2 166k
RD 2 3 4k
RL 3 0 20k
RS 5 4 0.5k

C1 6 8 INF
C2 3 0 INF
CS 5 0 INF

M1 3 6 5 5 NMOS

* NMOS with all terminals connected
* M1 Drain Gate Source Substrate NMOS

.end