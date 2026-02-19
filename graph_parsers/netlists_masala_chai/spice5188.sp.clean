spice
* SPICE Netlist

Vin 1 0 DC 0
R1 1 2 1k
R2 3 4 1k
Vcc 3 0 DC 15V
Vee 0 5 DC -15V
X1 2 0 3 5 opamp

* Subcircuit for opamp
.subckt opamp noninv inv out vcc vee
* Ideal opamp model
G1 out 0 value={V(noninv) - V(inv)}
Ec 3 0 value={Vcc}
Ev 5 0 value={Vee}
.ends opamp

.end