spice
* SPICE netlist for the given schematic

VCC 4 0 DC 15V
VEE 5 0 DC -15V
VIN1 3 0 DC 1V
VIN2 3 0 DC 1V

IBIAS1 3 5 DC 1mA
IBIAS2 2 5 DC 1mA
ILS 4 5 DC 1mA

Q1 2 3 5 NPN
Q2 4 2 2 NPN
Q3 4 2 5 NPN
Q4 6 4 5 NPN
Q5 5 6 6 NPN
Q6 4 2 5 NPN

D1 4 5 D
D2 5 6 D

RL 6 0 1k

.model D D
.model NPN NPN
.end