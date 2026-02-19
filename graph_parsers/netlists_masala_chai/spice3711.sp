plaintext
* SPICE Netlist for Circuit

VDD 5 0 DC 10V

R1 5 3 32k
R2 4 2 18k
RD 5 3 4k
RS 4 2 2k

M1 3 4 2 2 NMOS

.model NMOS NMOS (LEVEL=1)

.END