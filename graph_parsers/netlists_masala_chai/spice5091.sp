spice
* Op-Amp Circuit with Feedback
* Node assignments:
* 1 - Vout
* 2 - Ground
* 3 - Connection of Rf to Op-Amp Output
* 4 - Connection of R1, Rf, and Inverting Input
* 6 - AC Input Source

V1 6 2 AC 25mVp-p   ; Input AC voltage source
R1 6 4 180          ; Resistor R1 = 180 Ohms
Rf 3 4 1.8k         ; Feedback resistor Rf = 1.8k Ohms
X1 4 2 3 2 LF157A   ; Operational amplifier LF157A
* Connections for the op-amp: inverting input (4), non-inverting input (2), output (3), and ground (2 for dual supply assumption)