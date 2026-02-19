spice
* SPICE Netlist

V1 vj 0 DC VI
R1 vj 2 100k
R2 2 3 1k
R3 3 0 1k
R4 2 4 1k
R5 4 0 1k

D1 2 2 D
D2 3 3 D

VREF 5 3 DC V_REF
VREF_N 4 0 DC -V_REF

* Op-amp connections:
* Non-inverting input (+): node 2
* Inverting input (-): node 2
* Output: node 3

* End of netlist