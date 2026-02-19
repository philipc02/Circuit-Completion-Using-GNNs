spice
* Components
VDD VDD 4 DC 15
RD 2 4 1k
M3 3 2 0 0 NMOS L=1u W=2u
* Op-amp
E1 3 0 2 0 100k

* Nodes
* 0: Ground
* 2: Node connected to input of op-amp (A1) and gate of M3
* 3: Node connected to output of op-amp (A1) and drain of M3
* 4: Node between RD and VDD

* Voltage Sources
VDD 4 0 DC 15

* End of netlist