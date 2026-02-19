spice
* Netlist for the circuit

Vt 3 0 DC 0
RF 3 4 1k
RD 1 2 1k
M1 2 4 0 0 NMOS L=1u W=3u

* Define other model parameters if necessary
.model NMOS NMOS level=1

VDD 1 0 DC 5
VF 2 0

.end