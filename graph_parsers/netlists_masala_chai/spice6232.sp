spice
* Differential Amplifier Netlist
VCC VCC 0 DC 15
VEE VEE 0 DC -15

* Current Sources
I1 3 0 DC 1mA
I2 0 2 DC 1mA

* Transistors
Q1 3 1 2 NPN_model
Q2 2 1 2 NPN_model
Q3 4 3 3 PNP_model
Q4 0 4 2 PNP_model

* Resistor
RL 4 0 200

* Models (Assumptions made, replace with actual models if required)
.model NPN_model NPN
.model PNP_model PNP

* Simulation Commands
.TRAN 1u 100u
.END