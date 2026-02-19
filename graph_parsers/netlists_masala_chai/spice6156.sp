spice
* NMOS Transistors
M1 2 3 4 4 NMOS
M2 3 5 6 6 NMOS

* Current Sources
I1 2 7 DC 1mA
I2 2 0 DC 0.5mA

* Voltage Sources
Vsig 1 0 AC 1
Vo 5 0 DC 10

* Resistors
R1 1 3 10k
R2 5 6 10k

* Capacitors
C1 3 0 AC 1F

* Voltage Nodes
Vpos 7 0 DC 5
Vneg 6 0 DC -10

*.MODEL Statements for Transistors
.model NMOS NMOS LEVEL=1