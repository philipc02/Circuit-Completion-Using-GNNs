* Voltage Source
Vx 5 0 DC

* Resistors
Rpi 2 7 5.28k
Rf 2 3 82k
Rc 2 6 Rc_value

* Dependent Current Source
G1 2 4 2 7 18.9m

* Nodes
* 1: Non-designated node for analysis purposes
* 2: Common node for dependent source, resistors, and output
* 3: Resistor RF node
* 4: Dependent source control node
* 5: Vx positive terminal
* 6: Rc node
* 7: Ground

.END