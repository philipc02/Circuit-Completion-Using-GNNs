* Component Declaration
V1 5 0 DC 0
VI vi 5 SIN(0 1 1k)
IQ 2 9 DC 0.1A
M1 5 1 4 4 NMOS
C1 5 0 1u
C2 4 2 10u
CS 2 0 10u
RG 5 5 500k
RD 4 3 1k
RL 3 0 1k

* Nodes
* 1 - Drain of M1, one side of RD
* 2 - IQ, Top of CS
* 3 - Ground, other side of RD, RL
* 4 - Source of M1
* 5 - Gate of M1, one side of RG, C1

.model NMOS NMOS

.END