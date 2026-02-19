plaintext
* Components
* R3: Resistor between net 5 and ground
R3 5 3

* C2: Capacitor between net 2 and net 5
C2 2 5

* C1: Capacitor connected to net 3 and ground
C1 3 3

* R4: Resistor between net 6 and net 2
R4 6 2

* Opamp: Connected as a voltage follower
* Positive input at net 3, negative input at net 2
* Output at net 6
U1 3 2 6 opamp

* End of netlist