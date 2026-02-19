spice
* Op-amp Inverting Amplifier Circuit

V1 1 0 DC VI     * Input voltage source

R1 1 2 20k       * 20 kΩ resistor
R2 2 3 200k      * 200 kΩ resistor

XU1 0 2 2 opamp  * Op-amp with non-inverting input grounded

V2 3 0 DC VO     * Voltage source representing output

* Model for op-amp (ideal)
.subckt opamp 3 2 4
* Non-inverting, inverting, output
E1 4 0 3 2 100k
.ends opamp

.end