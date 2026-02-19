plaintext
* SPICE Netlist for the BJT Amplifier Circuit

* Transistors
Q1 3 4 5 NPN
Q2 2 6 7 NPN

* Resistors
RC 3 2 1000 ; Replace 1000 with the actual resistance value of Rc
RE 44 8 500 ; Replace 500 with the actual resistance value of Re

* Voltage Sources
VCC 7 3 DC 15 ; Replace 15 with the actual voltage of Vcc
VEE 8 44 DC -15 ; Replace -15 with the actual voltage of Vee

* AC Source
Vin 22 8 SIN(0 1 1k) ; Replace values with actual AC source parameters

* Output
Vout 2 8 DC 0

* Model Definitions
.model NPN NPN(IS=1E-15 BF=100)

.end