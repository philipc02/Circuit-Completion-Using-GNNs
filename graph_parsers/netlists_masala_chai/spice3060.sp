plaintext
* Transistor Definitions
M1  2 2 3 3 NMOS
M2  2 2 3 3 NMOS
M5  7 4 6 6 PMOS
M6  7 4 6 6 PMOS
M9  3 5 6 6 PMOS

* Current Source
I1  5 6 DC

* Resistors
Rro3_ro4  4 7 ro3 || ro4
RRon7_Ron8  6 5 Ron7 || Ron8

* Output Resistor Network
Rgmr012ro10  6 0 gm12 r012 r010/2

* Voltage Source for Vb0
Vb0  5 0 DC

* Common Mode Voltage Sources
Vout_CM1  6 0 DC Vout,CM
Vout_CM2  6 0 DC Vout,CM

.end