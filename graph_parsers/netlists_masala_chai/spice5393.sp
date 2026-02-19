spice
* SPICE netlist for Op-Amp Phase Shift Oscillator Circuit

* Operational Amplifier Subcircuit
* Example ideal Op-Amp; real implementations require specific models

* Amplifier Feedback Resistor
Rf 5 5 10k

* Phase Shift Network
C1 5 3 10n
R1 3 0 10k
C2 3 4 10n
R2 4 0 10k
C3 4 2 10n
R3 2 0 10k

* Input Voltage (assumed as reference)
V1 2 0 DC 0

* Output Voltage Node
Vout 7 0

.END