spice
* SPICE Netlist

* Current Source
I1 5 0 DC i₁

* Resistor r_d1
R1 6 0 r_d₁

* Voltage dependent current source
Gm1 3 0 Vgs gₘ₁

* Voltage dependent current source
Gm2 2 0 Vgs gₘ₂

* Resistor r_d2
R2 2 0 r_d₂

* Current Source
I0 2 4 DC i₀

* Connections to global ground reference
Vgs 3 0 DC 0

.end