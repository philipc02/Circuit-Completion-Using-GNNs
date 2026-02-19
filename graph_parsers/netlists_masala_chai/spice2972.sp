* SPICE Netlist

V_in 5 0 DC 0
V1 8 0 DC 0

RS 5 2 1k
RF 2 3 10k
RD 7 4 4.7k

I1 6 0 DC 1mA

* Node Assignments
* 1: Ground
* 2: Node between RS and RF
* 3: Node between RF and RD
* 4: Vout node
* 5: V_in+ terminal
* 6: Current source positive terminal
* 7: VDD node

.END