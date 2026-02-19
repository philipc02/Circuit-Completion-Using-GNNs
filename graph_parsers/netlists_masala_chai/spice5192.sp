plaintext
* Op-Amp Circuit

* Voltage Sources
VCC 2 0 DC VCC
VEE 2 0 DC -VEE

* Resistor
R1 vin 2 R

* Capacitor
C1 2 3 C

* Op-Amp
* Note: The op-amp is modeled as an ideal voltage-controlled voltage source (VCVS) with gain A_VOL
E1 3 0 2 2 A_VOL

* Connections
vin vin 0 DC 0
vout 3 0 DC 0

.end