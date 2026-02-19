spice
* SPICE Netlist for the given circuit

V1 in 0 DC 0

D1 net2 net2 D_model
R1 vi net2 1k
R2 net2 0 1k

* Diode model
.model D_model D

.END