spice
* BJT Amplifier Circuit

VCC 8 0 DC 10V

* Current Source
I_IS 9 3 DC

* Resistors
R1 5 2 17.9k
R2 2 3 1.4k
RC1 8 2 7k
RC2 8 7 2.2k
RE1 2 3 250
RE2 3 0 500
RF 4 3 5k
RL 7 0 2k

* Transistors
Q1 2 1 3 QMODEL1
Q2 7 6 3 QMODEL2

* Capacitors
C1 9 4
C2 5 0
C3 6 3

* .MODEL declarations
.model QMODEL1 NPN
.model QMODEL2 NPN

* End of netlist