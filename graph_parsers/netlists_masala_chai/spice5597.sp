spice
* Inverting Amplifier SPICE Netlist

Vi 1 0 DC 0V     * Input Voltage
R1 1 3 1k        * Resistor R1
C1 3 4 1u        * Capacitor C
R2 2 4 2k        * Resistor R2
XU1 0 4 2 Opamp  * Operational Amplifier
V1 0 2 DC 0V     * Voltage for Opamp

* Define op-amp (ideal model)
.model Opamp opamp(ad=1e6 gain=1e6)

.END