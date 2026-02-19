* SPICE Netlist

V1 1 0 DC 10V
V2 5 0 DC -10V

R_B 2 0 50k
R_C 4 3 10k
R_E 3 5 10k

Q1 4 2 3 NPN

* Node Mapping
* 1: +10V
* 2: Base of Q1, R_B
* 3: Emitter of Q1, R_E
* 4: Collector of Q1, R_C
* 5: -10V

.end