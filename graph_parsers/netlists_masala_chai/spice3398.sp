spice
* NMOS Amplifier Circuit

V1 5 0 DC VDD

RG1 5 4 RG1
RG2 4 6 RG2
RD 5 2 RD
RS 3 6 RS

* NMOS Transistor
M1 2 4 3 3 NMOS_MODEL

* Voltage Reference
VREF 4 0 DC VREF

.model NMOS_MODEL NMOS(Level=1)

.END