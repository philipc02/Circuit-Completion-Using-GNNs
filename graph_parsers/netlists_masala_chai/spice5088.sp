spice
* SPICE Netlist
Vin 5 0 AC 1
R1 5 2 1k
Rf 2 3 10k
RpRf 2 0 1k
XOP 0 2 3 opamp

.subckt opamp noninv inv out
* Ideal op-amp model
Vout out noninv dc 0
Iinv inv noninv dc 0
.ends opamp

.end