spice
* SPICE Netlist for the given schematic

V1 1 0 DC 0 ; Voltage Source V_T

* Transistors
Q1 2 6 8 QNPN ; NPN BJT model
Q2 4 5 8 QNPN ; NPN BJT model
Q3 7 8 8 QNPN ; NPN BJT model

* Resistors
RT 1 6 1000 ; RT resistor
RB11 3 2 100k ; RB11 resistor
RB12 3 4 100k ; RB12 resistor
RC1 3 5 2k ; RC1 resistor
RC2 3 7 2k ; RC2 resistor
RE1 5 8 200 ; RE1 resistor
RE2 8 8 200 ; RE2 resistor
RL 7 8 1k ; RL resistor

* Capacitors
C1 6 2 1u ; C1 capacitor
C2 5 4 1u ; C2 capacitor
C3 4 7 1u ; C3 capacitor
C4 7 8 1u ; C4 capacitor
C5 2 8 1u ; C5 capacitor
CC1 5 3 1u ; CC1 capacitor

.model QNPN NPN (BF=100)

.end