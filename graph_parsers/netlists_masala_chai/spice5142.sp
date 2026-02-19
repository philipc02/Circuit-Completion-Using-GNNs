plaintext
* SPICE Netlist for the Circuit

V1 1 0 DC Vin      * Input voltage source
R1 1 3 10k         * Resistor R1 with 10k ohms resistance
C1 3 4 1uF         * Capacitor C1 with 1 microfarad
XU1 3 0 2 out OPAMP * Op-Amp with inverting and non-inverting inputs

* Define Op-Amp subcircuit
.subckt OPAMP noninv inv out 
* Place your op-amp model details here 
* VCVS used for simplicity, actual models may vary
E1 out 0 VALUE = {V(noninv) - V(inv)}
.ends OPAMP

* Node mapping:
* 1 = Vin
* 2 = Ground
* 3 = R1 and non-inverting input of Op-Amp
* 4 = Capacitor C1 to Ground
* out = Output node of Op-Amp

.end