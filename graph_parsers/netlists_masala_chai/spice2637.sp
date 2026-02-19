spice
* NMOS Transistors
M1 3 3 5 5 NMOS
M2 6 2 5 5 NMOS

* Current Sources
I1 3 4 DC <value>
ISS 5 0 DC <value>

* Voltage Source
VDD 4 0 DC <value>

* Resistor
R1 4 2 <value>

* Voltage Labels
Vin 3 0 DC <Vin_value>
Vout1 3 0
Vout2 2 0
Vb 6 0

* Models (Assuming generic models are defined in the library)
.model NMOS NMOS

.END