* Voltage Input
Vi 3 4 DC 0
* Resistors
R1 2 3 1k
R2 5 0 1k
RL 6 10 1k
* Capacitor (approaches infinity, not included in simulation)
*C inf 2 3
* Voltage Source for VCC
VCC 1 0 DC 12V
* Dependent Current Source
G1 9 6 2 7 1
* Connections
* Node assignments: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
* DC Voltage Source and Ground
*.op
.end