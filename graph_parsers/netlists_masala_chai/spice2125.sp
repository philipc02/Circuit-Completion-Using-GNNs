spice
* NMOS Amplifier Circuit

* Voltage Source
VDD 5 0 DC 10

* Resistors
RD 5 2 1k
RS 2 0 100

* Capacitor
CS 2 0 1uF

* NMOS Transistor
M1 2 3 0 0 NMOS_MODEL

* Input Voltage Source
Vin 3 0 DC 0 AC 1

* Model definition for NMOS transistor
.model NMOS_MODEL NMOS
.end