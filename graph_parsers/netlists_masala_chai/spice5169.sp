spice
* SPICE Netlist for the Circuit

* Voltage Source
Vin 5 0 DC 0V

* Resistors
R1 5 3 R'
R2 3 2 R'
R3 2 4 R

* Capacitor
C1 3 2 C

* Operational Amplifier
* Assuming an ideal op-amp, using a default model OPAMP
XOP1 2 3 2 OPAMP

* Ground
Vg 4 0 DC 0V

.END