plaintext
* SPICE Netlist for the Circuit

* Resistors
RS 1 2 100k  ; Example resistance value
RD 3 4 10k   ; Example resistance value

* Capacitor
C1 2 3 1nF   ; Example capacitance value

* NMOS Transistor
M1 3 2 0 0 NMOS_MODEL  ; NMOS connected with drain at 3, gate at 2, source at 0

* Voltage Source
VDD 4 0 DC 5V ; VDD supplying 5 volts

* Model Definitions
.model NMOS_MODEL NMOS (LEVEL=1 KP=120u VTO=1)

* End of Netlist