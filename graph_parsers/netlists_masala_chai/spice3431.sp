spice
* Netlist for the given circuit

VDD 2 3 DC 5V
VT 7 5 DC 1V

RT 6 7 1k
RG 2 6 1k

CI 6 5 10p
CD 4 1 10p

QL 2 2 3 NMOS
QD 4 5 3 NMOS

.model NMOS NMOS (LEVEL=1)

* Node 3 is Ground
* Node 1 is VOUT

.END