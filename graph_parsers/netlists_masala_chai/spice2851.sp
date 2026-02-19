*MOSFET Amplifier Circuit

* NMOS Transistor
M1 3 Vin 0 0 NMOS

* Resistor
RD 3 VDD 3k

* Capacitor
CL 3 0 10p

* Voltage Source
VDD VDD 0 DC 5V
Vin Vin 0 DC 0V AC 1mV

* Model Definitions
.model NMOS NMOS (Level=1)

.end