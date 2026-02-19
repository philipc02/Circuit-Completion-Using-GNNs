*MOSFET definitions
M1 3 6 2 2 NMOS
M2 2 4 2 2 NMOS
M3 3 2 VDD VDD PMOS
M4 1 2 VDD VDD PMOS
M5 2 2 VDD VDD PMOS

*Current sources
I1 3 2 DC 1A
I2 2 VDD DC 1A

*Resistors
R1 5 2 1k
R2 3 1 1k
R3 2 VDD 1k

*Voltage source
VDD VDD 0 DC 5V

* Nodes
* 1: nA
* 2: A
* 3: Common source NMOS
* 4: Ground for Q2
* 5: Ground for Q1