plaintext
* NMOS Transistor
M1 2 Vin 0 0 NMOS

* PMOS Transistor
M2 VDD Vb 2 2 PMOS

* Voltage Source for VDD
VDD VDD 0 DC 1.8V

* Input Voltage Source
Vin Vin 0 DC 0V

* Define NMOS and PMOS with default parameters
.model NMOS nmos
.model PMOS pmos

.end