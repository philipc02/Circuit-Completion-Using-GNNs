spice
* SPICE Netlist for the circuit

VCC 2 0 DC <VCC>
VEE 3 0 DC <VEE>

RL1 2 7 <value_of_RL>
RL2 4 5 <value_of_RL>

Q1 7 10 6 NPN
Q2 5 8 6 NPN

IEE 6 3 DC <value_of_IEE>

* Voltage sources for biasing
V1 10 11 DC VALUE1
V2 8 9 DC VALUE2

* Ground connections
R1 11 0 1k
R2 9 0 1k

.end