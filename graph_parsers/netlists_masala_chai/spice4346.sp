spice
* Transistors
Q1 3 1 2 QNPN
Q2 4 3 2 QNPN

* Resistors
R1 3 5 (RC+DR/2)
R2 4 5 (RC-DR/2)
R3 2 0 RE

* Voltage Supply
V+ 5 0 DC V+
V- 0 2 DC V-

* Input
Vin1 1 0 DC V1
Vin2 4 0 DC V2

.model QNPN NPN
.END