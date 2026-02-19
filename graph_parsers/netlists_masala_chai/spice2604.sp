spice
* NMOS Transistors
M1 X Vin_CM 3 3 NMOS
M2 Y Vin_CM 3 3 NMOS

* Resistors
R1 X VDD RD
R2 Y VDD RD
RSS P 0 RSS

* Voltage Source
VDD VDD 0 DC 5

* Node Definitions
Vin_CM Vin_CM 0

* Analysis
*.op
.end