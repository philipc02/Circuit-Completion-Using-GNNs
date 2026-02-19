spice
* NMOS Amplifier Circuit
M1 2 3 0 0 NMOS
RD 1 2 1k
Cin 3 0 10u
VDD 1 0 5V
* Model for the NMOS
.model NMOS NMOS (Level=1 VTO=0.7 KP=120u W=1u L=1u)
.end