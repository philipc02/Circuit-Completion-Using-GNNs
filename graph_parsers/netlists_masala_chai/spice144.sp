spice
* SPICE Netlist for Differential Amplifier Circuit

* Transistors
Q1 4 7 2 BJT
Q2 4 8 2 BJT
Q5 6 4 2 BJT
Q6 9 5 2 BJT

* Current Source
IBIAS 9 2 DC 1mA

* Load Resistors
RL1 7 2 2k
RL2 8 2 2k

* Voltage Sources
VCC 9 0 DC 15V
VEE 2 0 DC -15V

.model BJT NPN (IS=1E-15 BF=100)
.END