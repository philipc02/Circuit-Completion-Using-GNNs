* Bipolar Junction Transistor Amplifier

* Voltage Sources
Vin 3 0 DC 0V
Vb 2 0 DC 0V
Vcc 4 0 DC 12V

* Resistors
RE 3 0 1k
RC 5 4 2k

* Transistors
Q1 6 3 3 NPN
Q2 5 2 6 NPN

* .MODEL Statement for NPN Transistors
.model NPN NPN (IS=1E-14 BF=200)

* Simulation Commands
.dc Vin 0 5 0.1
.end