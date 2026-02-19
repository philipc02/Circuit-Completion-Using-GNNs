spice
* SPICE Netlist

V1 4 1 DC Vd/2
RC 3 V+ 1k  ; Resistor connecting net 3 to V+
RB 4 3 1k   ; Resistor connecting net 4 to net 3
Q1 3 4 1 QNPN   ; NPN BJT with collector, base, emitter nodes

.model QNPN NPN (IS=1e-14 BF=200 VAF=100)

.end