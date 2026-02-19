spice
* Transistors
Q1 N2 Vb N2 NMOS
Q2 Vout N3 GND PMOS

* Current Sources
Iin N2 GND DC 1mA
Icc Vcc N4 DC 1mA

* Resistors
R1 Vout N2 1k
R2 N2 GND 1k
RF N2 Vb 1k

* Voltage Source
Vcc Vcc GND DC 5V

* Analysis
.op
.end