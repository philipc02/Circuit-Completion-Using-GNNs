* Op-Amp Circuit
VCC 1 0 DC 15V
VEE 0 5 DC -15V

* Op-Amp Model
XOP 2 2 3 1 5 OPAMP_MODEL

* Diode
D1 3 4 DIODE_MODEL

* Resistor
RL 4 0 1k

* Models
.model DIODE_MODEL D
.model OPAMP_MODEL OPAMP 

.end