spice
* Components
V1 3 0 DC 0
Rin 3 1 1k
R1 2 8 10k
R2 1 0 10k
RD 2 4 10k
RS 6 0 1k
RL 5 0 10k
CC1 3 1 1u
CC2 4 5 1u
M1 4 2 6 6 NMOS

* Voltage Source
VDD 8 0 DC 10V

* Simulation Commands
.tran 1ns 1us
.end