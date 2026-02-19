plaintext
* Spice netlist for given schematic
V1 Vin 0 DC 0

* Capacitor
C1 Vin N001 C=1uF

* Resistors
R3 N001 0 R=10k
R1 N002 0 R=1k
R2 N002 Vout R=100k

* Operational Amplifier
* (Assuming ideal op-amp)
XU1 N001 Vout N002 Vout opamp

* Voltage output
Vout N002 0

* End of netlist