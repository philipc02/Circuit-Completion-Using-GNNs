spice
* SPICE Netlist
* Operational Amplifier Circuit

* Components
R1 2 3 5k
R2 2 4 10k
VCC 23 2 DC 15V
VEE 4 2 DC -15V
Vin 3 6 SIN(0 1 50) * Sinusoidal input, adjust according to needs

* OPAMP Model (Using 318 Model)
XOPAMP 3 2 6 23 4 Opamp318

* Ground
GND 6 0

* Subcircuits
.subckt Opamp318 1 2 6 3 4
* ( non-inverting input, inverting input, output, VCC+, VEE- )
* Define the op-amp characteristics here
.ends Opamp318

* Analysis
.TRAN 1n 100u
.END