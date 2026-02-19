plaintext
* SPICE Netlist

VDD 2 0 DC 1.8
I1 1 2 DC 0.5m

M1 2 3 3 3 NMOS
M2 3 4 3 3 NMOS
M3 2 4 3 3 NMOS

* Node definitions
* 1: Current source node
* 2: VDD and M3 drain
* 3: M1 source and M2 source
* 4: Gate bias Vb

.END