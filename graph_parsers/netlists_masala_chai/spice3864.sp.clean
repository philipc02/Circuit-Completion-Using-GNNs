plaintext
* NPN BJT Amplifier Circuit

* Voltage Sources
VCC 6 0 DC <VCC_VALUE>
Vs 3 8 AC <VS_AMPLITUDE> 0

* NPN Transistor
Q1 2 4 5 NPN_MODEL

* Resistors
R1 2 10 <R1_VALUE>
R2 10 9 <R2_VALUE>
RC 6 2 <RC_VALUE>
RE 5 9 <RE_VALUE>

* Capacitor
CC 3 10 <CC_VALUE>

* Ground
0 9 8 0

* .MODEL statement for NPN (Example)
.model NPN_MODEL NPN(Is=1e-14 Bf=100)

* Control Statements
.control
tran 1n 10u
plot v(2) v(4)
.endc

* End of Netlist
.end