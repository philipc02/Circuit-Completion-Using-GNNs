plaintext
* Example SPICE netlist
V1 vcc 0 DC 5V
V2 vee 0 DC -5V

* Resistors
R1 vi 2 68k
R2 2 5 33k
R3 3 0 4.7k
R4 vcc 3 8.2k
R5 vcc 6 3.3k
R6 3 7 5.6k
R7 7 vo 2.4k

* Transistors
Q1 5 2 3 NPN
Q2 7 6 3 NPN
Q3 vo 7 0 NPN

* Voltage Inputs
vi vi 0 SIN(0V 0.1V 1kHz)

* Analysis
.TRAN 0.1us 10ms
.AC DEC 10 1 100k
.END