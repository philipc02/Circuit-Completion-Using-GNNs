plaintext
* SPICE netlist for the given circuit

V1 VDD 0 DC 5

M1 2 3 6 6 NMOS

C1 2 0 1u
CB 5 0 10u

R1 5 6 1k
R2 3 5 2k
R3 4 2 1k
R4 6 0 1k
RD 2 VDD 1k

.END