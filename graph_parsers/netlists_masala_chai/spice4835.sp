* Transistor
Q1 2 7 5 QNPN

* Voltage Sources
VCC 3 0 DC 10V
Vin 7 8 AC 1mV

* Resistors
R1 3 7 10k
R2 7 8 2.2k
Rc 3 2 3.6k
Re 5 8 1k
Rl 4 6 10k

* Capacitors
C1 2 4 1u  ; Assuming value for capacitor
C2 5 8 1u  ; Assuming value for capacitor

* Model for BJT
.model QNPN NPN