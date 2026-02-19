plaintext
* NMOS Transistor
M1 3 5 2 2 NMOS

* Voltage Source
V1 5 0 DC vi

* Resistor RG
RG 5 0 RG_VALUE

* Resistor RD
RD 6 3 RD_VALUE

* Current Source IQ
IQ 4 2 DC IQ_VALUE

* Capacitor CS
CS 2 0 CS_VALUE

* Voltage Source for RD
VDD 6 0 DC 5V

* Voltage Source for IQ
VSS 4 0 DC -5V

* .MODEL for NMOS
.model NMOS NMOS(L=1u W=1u)

.end