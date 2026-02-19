plaintext
* NMOS Transistors
M1 X A X X NMOS
M2 X nA X X NMOS

* Resistors
R1 X 5 R1
R2 Y X R2
R3 Y nA R3

* Voltage Source
V1 Vout 2 DC 0

* Op-Amp
* Using a generic op-amp model
.subckt OPAMP Y 2 Vout
* Connections:
* +  -  Out
X1 Y 2 Vout OPAMPMODEL
.ends OPAMP

* Connections to Op-Amp
XOPAMP Y 2 Vout OPAMP

* Model Definitions
.model NMOS nmos
.model PMOS pmos