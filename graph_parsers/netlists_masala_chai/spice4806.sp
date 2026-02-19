spice
* SPICE netlist for the given schematic

V1 3 6 DC 100uV
VCC 2 0 DC 10V
VEE 4 0 DC -2V

RB 3 2 2.7k
RC 2 5 3.6k
RE 5 4 1k
RL 2 7 100k

Q1 5 3 4 NPN

C1 2 0
C2 5 7

* DC analysis
.DC V1 -200u 200u 10u

* End of netlist
.END