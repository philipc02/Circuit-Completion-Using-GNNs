spice
* Operational Amplifiers
.subckt opamp in+ in- out
* Ideal opamp model, output feeds back into negative terminal
Vdiff in in- dc 0V
Eamp out 0 value = {V(in+, in-) * 1e6}
.ends opamp

* Circuit Description
* Voltage Source
V1 1 0 AC 1

* Capacitors
C1 1 2 10u
C2 2 22 10u
C3 22 3 10u

* Resistors
R1 2 0 10k
R2 4 5 100k
R3 4 5 100k

* Operational Amplifiers
X1 2 0 22 opamp
X2 22 0 3 opamp
X3 3 44 5 opamp