plaintext
* SPICE Netlist for Differential Amplifier
.model QNPN NPN (BF=100)

* NPN Transistors
Q1 3 6 5 QNPN
Q2 2 6 5 QNPN

* Resistors
RC1 6 6 1k ; Resistor between collector Q1 to Vcc
RC2 6 6 1k ; Resistor between collector Q2 to Vcc
R1 3 5 500 ; Resistor between base Q1 to emitter
R2 2 5 500 ; Resistor between base Q2 to emitter

* Current Source
I1 5 4 DC 1mA

* Voltage Source
VCC 6 0 DC 10V

* INPUT
Vin1 3 0
Vin2 2 0

* Output
Vout 1 0

.END