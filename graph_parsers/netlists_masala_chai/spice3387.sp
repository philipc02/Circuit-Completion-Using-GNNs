spice
* SPICE Netlist Generated from Schematics
* Resistor g_m (1 + g)
R1 0 6 g_m*(1+g)

* Current Source id = gm,lambda*(1+eta)*vgs
I1 3 2 dc g_m,lambda*(1+eta)*vgs

* .end statement to indicate the end of the SPICE deck
.end