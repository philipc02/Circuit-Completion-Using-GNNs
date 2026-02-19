* SPICE Netlist 

* Voltage Sources
V1 3 0 DC 0
VDD 4 2 DC VDD

* NMOS Transistor
M1 4 3 5 5 NMOS

* Capacitor
CH 4 2 CH

* Control Signals
VCK 5 0 PULSE

* Model Definitions
.model NMOS NMOS