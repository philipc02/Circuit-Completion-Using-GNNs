spice
* Op-Amp Circuit

* Voltage Source
Vin 1 2 AC 50m SIN(0 50m 1k)

* Operational Amplifier
XU1 2 0 2 3 741

* Resistors
Rf 2 3 3.9k
R1 2 0 100

* Power Supplies
Vcc 2 0 DC 15
Vee 3 0 DC -15

* Simulation Commands
.include '741.sub'  ; Include op-amp model
.end