plaintext
* SPICE netlist for the given circuit

V1 1 0 DC 8V
V2 4 0 DC 10V

RB 1 3 220k
RC 4 5 4k

Q1 5 3 2 NPN

.model NPN NPN (IS=1E-14 BF=100)