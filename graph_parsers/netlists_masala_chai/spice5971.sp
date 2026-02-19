plaintext
* NMOS Transistor (M1)
M1 2 3 6 6 NMOS

* Current Source
I1 2 5 DC 0.1mA

* Voltage-Controlled Current Source (VCCS)
G1 5 6 5 6 1

* Resistor (Rsig)
R1 5 6 10k

* Model Definitions
.model NMOS NMOS
.end