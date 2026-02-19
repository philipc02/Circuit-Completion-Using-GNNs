* SPICE Netlist

VIN 4 0 DC
R2 4 2 R2_value
R1 2 22 R1_value
ROUT 22 2 ROUT_value
E1 3 0 2 0 -A0
C1 2 0 C1_value

* Connections
* VIN: Node 4 to GND (0)
* R2: Between Node 4 and Node 2
* R1: Between Node 2 and Node 22
* ROUT: Between Node 22 and Node 2
* E1: Voltage-controlled voltage source with negative gain -A0, between nodes 3 to 0, controlled by nodes 2 to 0
* C1: Capacitor between Node 2 and GND (0)

.END