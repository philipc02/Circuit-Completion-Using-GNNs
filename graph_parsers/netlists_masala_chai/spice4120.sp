plaintext
* SPICE Netlist

Vi 4 0 DC 0
Rb 7 5 200
Rpi 5 3 2.5k
Cm 5 2 0.8p
Cp 5 6 10p
G1 3 6 5 6 0.04
Rl 2 3 2.5k

* Nodes
* 1: Unconnected
* 2: Connection point for Cμ and RL
* 3: Connection point for Rπ, G1, and RL
* 4: Positive terminal of Vi
* 5: Connection point for Rb, Rπ, and Cμ
* 6: Common ground
* 7: Node between Vi and Rb

.END