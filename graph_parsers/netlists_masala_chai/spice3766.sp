spice
* NMOS Transistor Circuit
M1 6 2 3 3 NMOS

* Resistor
RD 4 6 RD_value

* Voltage Sources
VDD 4 0 DC VDD_value
VI 2 0 DC VI_value

* Define MOSFET model
.model NMOS NMOS (kp=K_value Vto=Vt_value)

* Simulation Control
.tran 1n 100n
.end