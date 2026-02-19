plaintext
* SPICE Netlist for the given schematic

VCC 5 0 DC 5V
VEE 4 0 DC -5V
I1 VIN 3 DC 2mA

R1 5 5 1k
R2 5 5 1k
RL 2 VOUT 1

Q1 5 VIN 4 4 NPN
Q2 5 2 3 3 NPN

* Output stage is not defined in detail here
* DC bias at the input: VIN

.END