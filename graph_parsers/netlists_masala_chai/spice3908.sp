* Components
V1 1 2 DC 0
Vs 2 3 AC
R1 4 2 12k
R2 2 3 2k
RC 4 5 5k
RE 3 3 0.5k
CC 2 3

* Transistor
Q1 4 2 3 QMOD
.model QMOD NPN

* Voltage Sources
VCC 5 4 DC 5V
VEE 3 3 DC -5V

* Analysis
.AC DEC 10 1k 1Meg
.TRAN 1us 1ms
.END