spice
* NMOSFET M1
M1 Vout Vin GND GND NMOS W=10u L=0.18u

* PMOSFET M2
M2 VDD Vout Vout Vout PMOS W=22u L=0.18u

* Voltage Source
VDD VDD 0 DC 1.8V
Vin Vin 0 DC

* Simulation Options
.option post=2
.end