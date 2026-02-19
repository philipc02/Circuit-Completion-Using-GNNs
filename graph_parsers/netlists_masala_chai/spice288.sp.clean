* SPICE Netlist for given circuit

* Voltage Nodes:
* 5 (common node for Cf and R_L)
* 6 (connected to one side of gm)
* 8 and 9 (terminals for current source i_i)
* 2 (connected between R and Cin)
* 4 (termination for Vo)

* Resistors
R1 2 5 R       * Resistor R
RL 5 4 RL      * Load Resistor R_L

* Capacitors
Cin 2 0 Cin    * Capacitor C_in
Cf 5 6 Cf      * Capacitor C_f

* Current Sources
Iinput 8 9 Ii   * Input current source i_i
Gm 6 0 5 6 Gm   * Voltage dependent current source g_m v_1

* Assign the input current source from nodes 8 to 9