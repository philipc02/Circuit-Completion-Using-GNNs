plaintext
* Op-Amp Configuration
X1 vin 4 vout VCC VEE opamp

* Diode D1
D1 4 0 Dmodel

* NPN Transistor Q1
Q1 4 vout 0 NPNmodel

* Voltage Sources
VCC VCC 0 DC 15V
VEE VEE 0 DC -15V

* Models
.model Dmodel D
.model NPNmodel NPN