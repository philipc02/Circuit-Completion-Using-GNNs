spice
* Op-Amp Circuit SPICE Netlist

* Voltage Sources
VCC 7 0 DC 15V
VEE 4 0 DC -15V

* Input Source
Vin 2 0 DC 0V

* Resistors
R1 2 0 1.5k
R2 4 3 68k

* Operational Amplifier
XOAmp 2 3 6 7 4 OPAMP_MODEL

* Output
Vout 6 0

* Model Definition for Op-Amp
.subckt OPAMP_MODEL 2 3 6 7 4
* Idealized Op-Amp Model
.ends OPAMP_MODEL

.control
tran 1n 10u
plot v(6)
.endc
.end