* NMOS and PMOS transistors
M1 3 Vin 2 2 PMOS
M2 3 4 4 4 NMOS

* Voltage source
VDD 2 0 DC VDD

* Input voltage source
Vin Vin 0 DC Vin

* .MODEL statements for transistors (Example, values need to be defined)
.model PMOS PMOS (KP=120u VTO=-0.7)
.model NMOS NMOS (KP=100u VTO=0.7)

* Analysis commands
*.dc Vin 0 5 0.1
*.tran 1n 100n
.end