plaintext
* SPICE Netlist for the given circuit

M1 4 2 3 3 NMOS

R1 5 2 10k
R2 2 0 5k

VDD 5 0 DC 5V

* Define NMOS transistor model
.model NMOS NMOS (LEVEL=1)

.END