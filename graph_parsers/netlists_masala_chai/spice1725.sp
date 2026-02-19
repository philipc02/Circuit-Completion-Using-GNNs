plaintext
* SPICE Netlist
VDD 44 0 DC 5V
Vin 1 0 DC 0V

Rs 1 2 1k
RD 44 4 1k
RP 3 0 1k
Cin 2 3 1u
CL 4 5 1u

M1 4 2 3 3 NMOS L=1u W=1u

* model definition
.model NMOS NMOS LEVEL=1

.end