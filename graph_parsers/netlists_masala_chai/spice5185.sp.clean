spice
* Op-Amp Circuit

* Voltage Sources
VCC 3 0 DC 15V
VS 9 0 DC 5V
VIN 5 0 DC vin_value * replace vin_value with actual input voltage

* Resistors
R1 5 4 R1_value * replace R1_value with actual resistance
R2 4 0 R2_value * replace R2_value with actual resistance
R3 2 9 1k

* Op-Amp
* (339 model not detailed, assuming ideal for simulation)
XOPAMP 5 4 2 3 12 opamp_model
.model opamp_model opamp

* Connections
* 5 - Non-inverting input, 4 - Inverting input, 2 - Output
* 3 - VCC, 12 - Ground

.end