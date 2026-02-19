plaintext
* NMOS Transistor
M1 2 3 5 5 NMOS_Model

* Resistor
RD 2 1 10k

* Capacitor
CDB 2 7 1p

* Voltage Source
VDD 1 0 DC 5V

* Input
VIN 3 0 SIN(0 1V 1kHz)

* Models
.model NMOS_Model NMOS (Level=1)