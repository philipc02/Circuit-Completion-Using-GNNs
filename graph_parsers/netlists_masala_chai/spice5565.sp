plaintext
* SPICE Netlist

* Components
V1 1 0 DC Vi1
R1 1 3 10k
R2 3 2 10k
R3 2 0 10k
R4 2 0 10k

* Op-amp
* Assume ideal op-amp with non-inverting terminal at net 4 and inverting terminal at net 3
U1 4 3 2 2 OPA

* Analysis
.tran 0 1m

.end