spice
* NMOS Amplifier Circuit

* Transistor
M1 Vout Vin 0 0 NMOS

* Current Source
I1 I+ 0 1mA

* Resistors
R1 I+ Vout r_o3
R2 Vout 0 r_o2

* Parameters (example values, replace as needed)
.model NMOS nmos (level=1 Vto=0.7 Beta=2e-3)