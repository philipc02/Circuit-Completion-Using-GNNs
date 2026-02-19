spice
* Transistors
Q1 4 3 2 NPN
Q2 5 6 2 NPN

* Resistors
R1 6 4 47k ; RC (left)
R2 2 5 47k ; RC (right)
R3 2 0 68k ; RE

* Voltage Sources
V1 1 0 DC 15 ; VCC
V2 2 0 DC -15 ; VEE
V3 3 0 SINE(0 5m 1k) ; v1

* Node assignments
* 1: VCC
* 2: VEE
* 3: Input to Q1 Base
* 4: Collector of Q1
* 5: Collector of Q2
* 6: Vout

.end