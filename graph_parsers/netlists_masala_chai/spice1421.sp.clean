plaintext
* SPICE Netlist

* PMOS Transistor
M1 3 N001 N002 PMOS 

* NMOS Transistor
M2 N002 N002 0 NMOS

* Resistor
Rout N001 0 ROUT

* Voltage Sources
Vb1 3 0 DC 1.8V
Vb2 2 0 DC 0V

.model PMOS PMOS (KP=30u VTO=-0.7)
.model NMOS NMOS (KP=120u VTO=0.7)

*.end