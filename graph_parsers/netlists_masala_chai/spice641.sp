spice
* Diode Bridge with Load Resistor

Vin 1 0 DC 0

* Diodes
D1 2 3 D_model
D2 3 4 D_model
D3 6 7 D_model
D4 6 9 D_model

* Resistor
RL 5 8 1000

* Voltage Source
Vout 5 0

.model D_model D

.end