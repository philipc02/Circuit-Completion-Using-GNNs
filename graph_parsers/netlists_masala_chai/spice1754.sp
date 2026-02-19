plaintext
* SPICE Netlist for the Amplifier Circuit

VCC VCC 0 DC 10V
VIN VIN 0 DC 0V

RC 2 VCC 1000
RB 3 VIN 1000

Q1 2 3 0 NPN

.model NPN NPN

.END