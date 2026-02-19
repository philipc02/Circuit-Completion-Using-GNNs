* Transistor Q1
Q1 2 2 3 NPN

* Transistor Q2
Q2 5 4 6 NPN

* Current Source
IBias 3 0 DC 1mA

* Resistor
Ro 5 0 1k

* Simulation Commands
.model NPN npn (Is=1e-14 Bf=100)
.options post
.end