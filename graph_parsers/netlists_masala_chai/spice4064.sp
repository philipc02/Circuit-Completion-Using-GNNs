spice
* SPICE netlist for the given schematic
.param gm=1m

* Voltage sources
Vgs 7 3 DC Vgs

* Capacitors
Cgd 7 2 Cgd_value
Cgs 3 7 Cgs_value
Cds 2 0 Cds_value

* Voltage-controlled current source
Gm 3 4 Vgs gm

* Resistors
Rs 3 0 Rs_value
Ro 3 4 Ro_value
Rd 4 2 Rd_value

* End of netlist