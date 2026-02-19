spice
* Differential Pair Circuit
* Node 0 is considered to be ground.

* Voltage and Current Sources
VCC 5 0 DC VCC
I1 3 0 DC I

* BJTs
Q1 5 2 2 NPN
Q2 4 2 2 NPN

* Resistors
Rc1 5 4 Rc
Rc2 4 5 Rc
Re1 2 3 Re
Re2 2 3 Re

* Definitions
.model NPN NPN

.END