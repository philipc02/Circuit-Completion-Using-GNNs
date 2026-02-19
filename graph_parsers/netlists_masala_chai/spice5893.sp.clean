spice
* SPICE netlist for the given circuit

V1 5 0 AC 1
I1 3 0 DC 0.5mA
R1 3 7 200k
R2 5 2 200
R3 4 0 20k
C1 5 7 AC 1uF
C2 3 4 AC 1uF

* Assume the capacitor is 1uF for simulation purposes, even though ∞ signifies a short in AC.
Q1 3 7 2 NPN

VIN 5 0

* Define NPN model
.model NPN NPN (IS=1e-15 BF=100)

.end