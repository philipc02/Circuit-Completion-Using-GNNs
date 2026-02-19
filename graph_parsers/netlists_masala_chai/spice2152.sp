spice
* Netlist for the given schematic
M1 4 2 3 3 NMOS
RD 4 VDD 1k
Vx 3 0 DC 1V
Ix 3 DeltaV DC 1A

* Define nodes
* 2 = Vb
* 3 = common node for Ix and source of M1
* 4 = connection between RD and drain of M1
* VDD = Supply voltage node