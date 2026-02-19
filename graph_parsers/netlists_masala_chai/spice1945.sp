spice
* Components
I1 4 2 DC 1A
Q1 2 1 3 NPN
Q2 3 5 5 NPN
D1 1 5 D
D2 5 5 D
RL 3 0 1k

* Voltage Sources
VCC 4 0 DC 10V
VEE 5 0 DC -10V
Vin 1 5 AC 1V

* Nodes
* 1 = Vin
* 2 = Connection between I1 and Q1 base
* 3 = Connection between Q1 collector, Q2 collector and RL
* 4 = VCC
* 5 = VEE and ground reference

* Model specifications
.model D D
.model NPN NPN (IS=1E-14 BF=100)