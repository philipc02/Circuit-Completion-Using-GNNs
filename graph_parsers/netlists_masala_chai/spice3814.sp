spice
* SPICE Netlist

* Current Source
I1 8 0 DC 0

* Resistors
R1 8 6 100k
RS 6 0 0
RG 3 2 50k
RD 3 4 0
RL 5 0 2k

* Capacitors
C1 3 6 C1_value
C2 5 4 C2_value
CG 2 0 CG_value

* Voltage Sources
Vplus 6 0 DC 5
Vminus 3 2 DC -5

.end