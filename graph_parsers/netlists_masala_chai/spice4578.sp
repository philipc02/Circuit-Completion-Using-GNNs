spice
* CMOS Inverter Pair Netlist
*.LIB 'your_model_file.lib' TT

VDD 5 0 DC 5V
VI 8 0 DC

* PMOS Transistors
MP1 2 9 5 5 PMOS
MP2 7 2 5 5 PMOS

* NMOS Transistors
MN1 2 9 3 3 NMOS
MN2 7 2 10 10 NMOS

* Input and Outputs
* VI is the input voltage source connected to net 8
* VO1 is the output voltage at net 2
* VO2 is the output voltage at net 6

* Ground connections
VSS 0 0

.END