spice
*MOSFET Circuit
* Nodes: 1 = v1, 2 = V+, 3 = Output node (vO, v2), 4 = M1 Source, 5 = RL Ground, 6 = IQ Current Source, 7 = RD to V+

* Transistors
M1 2 1 4 4 NMOS
M2 3 3 4 4 NMOS

* Resistors
RD 2 7 1k
RL 3 5 1k

* Current Source
IQ 4 6 DC 1mA

* Voltage Sources (for simulation purposes, assumptions)
V1 1 0 DC 1V
V2 2 0 DC 5V
V3 6 0 DC -5V