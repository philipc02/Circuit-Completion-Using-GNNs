* SPICE Netlist

M1 2 Vb 4 4 NMOS    * NMOS transistor with Drain=2, Gate=Vb, Source=4, Bulk=4
RD 3 2 RD_value     * Resistor RD with positive terminal=3, negative terminal=2
Iin 4 0 DC Iin_value * Current source Iin connected between node 4 and ground
I1 5 0 DC I1_value  * Current source I1 connected between node 5 and ground
Cp 4 5 Cp_value     * Capacitor CP connected between node 4 and node 5
VDD 3 0 DC VDD_value * Voltage source VDD connected between node 3 and ground

* .model statements and component values can be added here
.model NMOS NMOS