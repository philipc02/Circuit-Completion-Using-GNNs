spice
* MOSFET Amplifier Circuit
M1 Vout Vin 2 2 NMOS
RD Vdd Vout 1k
RS 2 0 500
Vin Vin 0 DC 1.0V
Vdd Vdd 0 DC 5.0V
* NMOS model parameters (example)
.model NMOS NMOS (Level=1 VTO=0.7 KP=120u)
.END