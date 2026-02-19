plaintext
* DC Supply Voltages
VCC 7 0 DC 15V
VEE 8 0 DC -15V

* AC Input Source
V1 3 0 AC 1mV

* Resistors
RC1 2 7 5k
RC2 2 6 5k
RE 4 8 7.5k

* NPN Transistors
Q1 2 3 4 NPN
Q2 2 5 4 NPN

* Simulation Commands
.ac dec 10 1 100k
.end