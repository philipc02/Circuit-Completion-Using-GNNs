* NMOS with body tied to source
M1 Vout Vin 2 2 NMOS

* Resistors
RD VDD Vout 10k
R2 VDD 2 10k
R1 2 0 10k
RS Vin 0 10k

* Voltage supply
VDD VDD 0 DC 1.8V

* Model for NMOS (replace with actual model parameters)
.model NMOS NMOS (KP=120u VT0=0.7 GAMMA=0.4 PHI=0.6 LAMBDA=0.02)

* Input/output pins
Vin Vin 0
Vout Vout 0

.end