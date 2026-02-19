plaintext
* Netlist for the given circuit

VDD 2 0 DC 1.8V

R1 2 4 1k
R2 2 0 10k
R3 2 0 20k
R4 3 0 200
RP 4 3 1k

M1 4 2 3 3 NMOS

.model NMOS NMOS Level=1

.END