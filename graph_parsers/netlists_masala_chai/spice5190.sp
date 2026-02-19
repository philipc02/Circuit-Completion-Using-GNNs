plaintext
* SPICE Netlist
* Op-amp A1 with 6 as non-inverting, 3 as inverting input, 2 as output
X1 6 3 2 OPAMP

* Op-amp A2 with 3 as non-inverting, 2 as inverting input, 3 as output
X2 3 2 3 OPAMP

* Diode D1 
D1 2 4 DIODE_MODEL

* Diode D2
D2 3 5 DIODE_MODEL

* Resistor RL
RL 5 7 1k

* Voltage Sources
VCC 4 0 DC +VCC
VEE 0 0 DC -VEE