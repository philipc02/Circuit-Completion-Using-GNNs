* SPICE Netlist

VDD 5 0 DC 5V
VIN 3 0 DC 1V

RD 5 4 1k

M1 4 2 3 3 NMOS

* Node Assignments:
* 1. M1 Drain -> Node 4
* 2. M1 Gate -> Node 2 (connected to Vb)
* 3. M1 Source -> Node 3 (connected to Vin)

.END