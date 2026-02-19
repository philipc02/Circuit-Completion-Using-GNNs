plaintext
* SPICE Netlist for given circuit

VDD 3 0 DC 1.8V

R1 3 4 1k
RD 3 2 1k
R2 4 0 1k

M1 2 4 0 0 NMOS_MODEL

* NMOS Model
.model NMOS_MODEL NMOS (LEVEL=1)

.end