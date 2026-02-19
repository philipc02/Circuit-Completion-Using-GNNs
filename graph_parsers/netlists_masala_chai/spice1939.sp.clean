spice
* SPICE Netlist for BJT Circuit

* NPN transistor Q1
Q1 4 5 3 NPN_MODEL

* PNP transistor Q2
Q2 2 4 1 PNP_MODEL

* Load Resistor
RL 3 0 1k

* Voltage Sources
VCC 5 0 DC 10V
VEE 1 0 DC -10V

* Continuation of the circuit design
VOUT 3 0

.model NPN_MODEL NPN (IS=1e-16 BF=100)
.model PNP_MODEL PNP (IS=1e-16 BF=100)

.END