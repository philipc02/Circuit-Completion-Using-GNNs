spice
* Nodes
* 1 : vI
* 2 : node connecting IBias, M3
* 3 : node connecting M4
* 6 : output (vO) and RL
* 7 : V+
* 8 : output (vO) to RL
* 9 : drain of M3

* Voltage sources
V+ 7 0 DC 10V
V- 0 0 DC -10V

* Current Source
I_bias 7 2 DC I_bias_value

* Transistors
M1 7 2 6 6 PMOS
M2 6 3 0 0 NMOS
M3 2 9 7 7 PMOS
M4 0 3 3 3 NMOS

* Resistor
RL 6 8 RL_value

* Simulation commands
.op
.end