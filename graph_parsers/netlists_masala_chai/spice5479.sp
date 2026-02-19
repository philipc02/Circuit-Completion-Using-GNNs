spice
* SPICE netlist for the given schematic

VCC 1 0 DC VCC
VEE 4 0 DC -VCC
Vin 6 0 AC Vin

Q1 1 2 3 NPN
Q2 3 2 4 NPN

RL 3 4 RL

* Connections
* Node 2 connected between bases of Q1 and Q2, and via input voltage source
* Node 3 connected between emitters of Q1 and Q2, and collector of Q2
* Node 1 connected to +VCC
* Node 4 connected to -VCC

.END