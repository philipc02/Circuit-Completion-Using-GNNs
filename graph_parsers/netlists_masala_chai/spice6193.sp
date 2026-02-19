spice
* SPICE netlist for the given circuit

VDD 9 0 DC VDD
Is 8 3 DC Is

RD 9 6 RD
Rin 2 3 Rin
RF 8 2 RF
RL 5 2 RL
RM 2 0 RM

Q1 9 7 8 NMOS
Q2 5 4 6 PMOS

* Models
.model NMOS NMOS(...)
.model PMOS PMOS(...)

.END