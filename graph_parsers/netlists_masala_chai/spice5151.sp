plaintext
* Netlist for the provided schematic
R1 Vin N2 R
R2 N2 N3 R
C1 N2 0 C1
C2 N3 N2 C2
* Op-amp model
* Assuming an ideal op-amp with very high gain
* Note: SPICE requires an op-amp model. Here we place pins.
XU1 N2 N2 Vout opamp
* Voltage input
Vin Vin 0 DC 0V
* The following should be defined elsewhere or use a specific model:
* .model opamp opamp(V(olmax)=1e5)
.end